"""
Lectus Creative Engine — Telegram Bot

No commands. Just talk naturally.
Send photos → AI learns your taste.
Ask for anything → Claude decides what to do using tool-use.
"""

import asyncio
import json
import logging
import tempfile
from dataclasses import replace
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import anthropic
import config
from brain.client import Brain
from taste.vision import analyze_inspiration, format_analysis_for_telegram
from taste.feedback import parse_feedback, format_feedback_response
from taste.memory import get_taste_summary
from taste.drive_watcher import DriveWatcher
from taste.template_builder import build_templates, format_templates_summary, load_templates_from_brain
from pipeline.types import PipelineInput
from pipeline.orchestrator import run_pipeline, run_carousel, format_result_for_telegram
from pipeline.steps.render import render as render_post
from pipeline.steps.image_gen import generate_image
from pipeline.steps.brain_read import brain_read
from pipeline.steps.dynamic_template import (
    save_liked_template, save_client_preference,
    get_client_preferences, generate_dynamic_template,
)
from pipeline.types import CreativeDecisions, ImageResult

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────
brain = Brain(url=config.TURSO_DATABASE_URL, auth_token=config.TURSO_AUTH_TOKEN)
drive_watcher = DriveWatcher(brain)
_ai_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Track analyses for feedback
_last_analysis_by_user: dict[int, dict] = {}
_analysis_by_msg: dict[int, dict] = {}

# Conversation history per user — stores full API message objects for tool-use
MAX_HISTORY = 20
_chat_history: dict[int, list[dict]] = {}

# Track last pipeline result per user for edits — persisted to disk
_last_post_by_user: dict[int, dict] = {}
_LAST_POST_FILE = Path(config.OUTPUT_DIR) / ".last_posts.json"

# Track previous decisions per user for variety
_previous_decisions: dict[int, list[dict]] = {}


def _save_last_posts():
    """Persist last post data to disk so it survives restarts."""
    try:
        # Serialize — skip non-JSON-serializable fields, keep what we need for resend
        serializable = {}
        for uid, data in _last_post_by_user.items():
            d = data.get("decisions")
            serializable[str(uid)] = {
                "client": data.get("client", ""),
                "rendered_path": data.get("rendered_path", ""),
                "template_html": data.get("template_html", ""),
                "logo_b64": data.get("logo_b64"),
                "decisions": {
                    "headline": d.headline, "subtext": d.subtext, "cta": d.cta,
                    "template": d.template,
                    "font_headline": d.font_headline, "font_headline_weight": d.font_headline_weight,
                    "font_headline_size": d.font_headline_size,
                    "font_subtext": d.font_subtext, "font_subtext_weight": d.font_subtext_weight,
                    "color_bg": d.color_bg, "color_text": d.color_text,
                    "color_accent": d.color_accent, "color_subtext": d.color_subtext,
                } if d else None,
                "image_path": data.get("image", {}).image_path if data.get("image") else None,
                "image_url": data.get("image", {}).image_url if data.get("image") else None,
            }
        _LAST_POST_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_POST_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save last posts: {e}")


def _load_last_posts():
    """Restore last post data from disk after restart."""
    global _last_post_by_user
    try:
        if _LAST_POST_FILE.exists():
            data = json.loads(_LAST_POST_FILE.read_text())
            for uid_str, post_data in data.items():
                uid = int(uid_str)
                d = post_data.get("decisions")
                if d:
                    decisions = CreativeDecisions(
                        headline=d.get("headline", ""), subtext=d.get("subtext", ""),
                        cta=d.get("cta", ""), template=d.get("template", ""),
                        font_headline=d.get("font_headline", "Inter"),
                        font_headline_weight=d.get("font_headline_weight", "800"),
                        font_headline_size=d.get("font_headline_size", 68),
                        font_subtext=d.get("font_subtext", "DM Sans"),
                        font_subtext_weight=d.get("font_subtext_weight", "400"),
                        color_bg=d.get("color_bg", "#F5F4F0"),
                        color_text=d.get("color_text", "#2A2A2A"),
                        color_accent=d.get("color_accent", "#C4A77D"),
                        color_subtext=d.get("color_subtext", "#6B6B6B"),
                    )
                    image = ImageResult(
                        image_path=post_data.get("image_path", ""),
                        image_url=post_data.get("image_url", ""),
                        model_used="restored", prompt_used="",
                    ) if post_data.get("image_path") else None
                    _last_post_by_user[uid] = {
                        "decisions": decisions,
                        "image": image,
                        "client": post_data.get("client", ""),
                        "template_html": post_data.get("template_html", ""),
                        "logo_b64": post_data.get("logo_b64"),
                        "rendered_path": post_data.get("rendered_path", ""),
                    }
            logger.info(f"Restored last posts for {len(_last_post_by_user)} users from disk")
    except Exception as e:
        logger.warning(f"Failed to load last posts: {e}")


def _add_to_history(user_id: int, role: str, content):
    """Add a message to user's conversation history.
    Content can be a string or a list of content blocks (for tool_use)."""
    if user_id not in _chat_history:
        _chat_history[user_id] = []
    _chat_history[user_id].append({"role": role, "content": content})
    # Trim but never split tool_use/tool_result pairs
    _trim_history(user_id)


def _trim_history(user_id: int):
    """Trim history to MAX_HISTORY, never splitting tool_use from tool_result."""
    hist = _chat_history.get(user_id, [])
    if len(hist) <= MAX_HISTORY:
        return
    # Find the safest trim point: skip forward until we're not in a tool_result
    trim_to = len(hist) - MAX_HISTORY
    while trim_to < len(hist):
        msg = hist[trim_to]
        # Don't start on a tool_result (it needs the preceding assistant tool_use)
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            # Check if it's a tool_result block
            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in msg["content"]):
                trim_to += 1
                continue
        break
    _chat_history[user_id] = hist[trim_to:]


def _get_history_for_api(user_id: int) -> list[dict]:
    """Get conversation history formatted for the Anthropic API.
    Returns the messages list, ensuring it starts with 'user' role."""
    hist = _chat_history.get(user_id, [])
    if not hist:
        return []
    # Ensure starts with user message
    start = 0
    for i, msg in enumerate(hist):
        if msg["role"] == "user":
            start = i
            break
    return hist[start:]


# ── Tool Definitions ──────────────────────────────────────

TOOLS = [
    {
        "name": "generate_post",
        "description": (
            "Generate a brand new social media post from scratch with a new concept and image. "
            "Use ONLY when the user wants something completely new — a new idea, new concept, new design. "
            "Do NOT use this for modifications to an existing post (use edit_post instead)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {
                    "type": "string",
                    "description": "Client/brand name (e.g. 'LMW', 'Georgoulis', 'Somamed'). Use 'ALL' if not specified.",
                },
                "brief": {
                    "type": "string",
                    "description": "Creative direction, theme, or topic for the post. Be specific.",
                },
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "instagram", "facebook"],
                    "description": "Social media platform. Default: linkedin",
                },
                "image_source": {
                    "type": "string",
                    "enum": ["auto", "stock", "ai"],
                    "description": "Image source preference. 'stock' = stock photos only, 'ai' = AI-generated only, 'auto' = try stock first then AI. Default: auto. Use 'stock' when user explicitly asks for stock/real photos.",
                },
                "use_last_inspiration": {
                    "type": "boolean",
                    "description": "Set to true ONLY when the user explicitly asks to make a post 'like this', 'similar to this', 'based on this' referring to an inspiration image they just sent. This forces the template to replicate that specific layout. Do NOT set this when generating a normal post.",
                },
            },
            "required": ["client", "brief"],
        },
    },
    {
        "name": "generate_carousel",
        "description": (
            "Generate multiple cohesive posts as a carousel/series with the same design language. "
            "Use when the user asks for multiple posts, a carousel, a series, or slides."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client/brand name"},
                "brief": {"type": "string", "description": "Theme/direction for the carousel"},
                "count": {
                    "type": "integer",
                    "description": "Number of posts (default 6, max 10)",
                    "minimum": 2,
                    "maximum": 10,
                },
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "instagram", "facebook"],
                },
            },
            "required": ["client", "brief"],
        },
    },
    {
        "name": "edit_post",
        "description": (
            "Modify the last generated post — change text, colors, fonts, sizes, layout, translate text, "
            "add/remove logo. The hero image stays the same. Use for ANY tweak to the existing post: "
            "translating, restyling, changing text, adjusting layout. "
            "IMPORTANT: Only include fields that need to change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "New headline text (change for translation or rewording)"},
                "subtext": {"type": "string", "description": "New supporting text"},
                "cta": {"type": "string", "description": "New call-to-action text"},
                "template": {
                    "type": "string",
                    "enum": ["object-hero", "text-dominant", "split", "full-bleed"],
                    "description": "Layout template. Only change if user explicitly wants different layout.",
                },
                "font_headline": {"type": "string", "description": "Google Font name"},
                "font_headline_weight": {"type": "integer", "description": "Font weight (400-900)"},
                "font_headline_size": {"type": "integer", "description": "Font size in pixels (40-120)"},
                "font_headline_tracking": {"type": "string", "description": "Letter spacing CSS value"},
                "font_headline_case": {
                    "type": "string",
                    "enum": ["uppercase", "lowercase", "none"],
                },
                "color_bg": {"type": "string", "description": "Background color (hex)"},
                "color_text": {"type": "string", "description": "Text color (hex)"},
                "color_accent": {"type": "string", "description": "Accent color (hex)"},
                "color_subtext": {"type": "string", "description": "Subtext color (hex)"},
                "headline_margin_x": {"type": "integer", "description": "Horizontal margin (0-200px)"},
                "headline_margin_y": {"type": "integer", "description": "Vertical margin (0-200px)"},
                "headline_max_width": {"type": "string", "description": "Max width CSS (e.g. '75%')"},
                "image_padding": {"type": "integer", "description": "Image padding (0-200px)"},
                "add_logo": {"type": "boolean", "description": "Set true to add the client's logo"},
                "remove_logo": {"type": "boolean", "description": "Set true to remove the logo"},
            },
            "required": [],
        },
    },
    {
        "name": "replace_image",
        "description": (
            "Replace the hero image in the last post while keeping all design decisions (font, colors, layout, text). "
            "Searches for a new stock photo or generates a new AI image using the same concept. "
            "Use when user says 'replace the photo', 'use a stock image', 'change the picture', etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_source": {
                    "type": "string",
                    "enum": ["auto", "stock", "ai"],
                    "description": "Image source preference. 'stock' = stock photos only, 'ai' = AI-generated only, 'auto' = try stock first then AI. Use 'stock' when user asks for stock/real photos.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "save_favorite",
        "description": (
            "Save the current post design as a favorite/liked template. Use when the user approves, "
            "loves, or wants to save the current post. Can include modifications to save alongside."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client to save for (overrides post's client)"},
                "modifications": {
                    "type": "object",
                    "description": "Design tweaks to save (e.g. {'color_accent': '#FF6B00'})",
                },
            },
            "required": [],
        },
    },
    {
        "name": "process_feedback",
        "description": (
            "Process user feedback on the last taste analysis of an inspiration image. "
            "Use when user confirms, corrects, or refines observations about a photo they sent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback_text": {"type": "string", "description": "The user's feedback on the analysis"},
            },
            "required": ["feedback_text"],
        },
    },
    {
        "name": "get_taste_profile",
        "description": (
            "Show what the system has learned about the user's design taste and preferences. "
            "Use when user asks about their taste, style, preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "manage_templates",
        "description": "Show existing templates or rebuild them from taste data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["show", "rebuild"],
                    "description": "'show' to list templates, 'rebuild' to regenerate from taste",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "resend_last_post",
        "description": "Re-send the last generated post image. Use when user says 'send it', 'show me', etc.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ── System Prompt Builder ─────────────────────────────────

def _build_system_prompt(user_id: int) -> str:
    """Build the system prompt with current state context."""
    has_analysis = user_id in _last_analysis_by_user
    has_post = user_id in _last_post_by_user
    post_data = _last_post_by_user.get(user_id, {})

    post_context = ""
    if has_post and post_data.get("decisions"):
        d = post_data["decisions"]
        post_context = (
            f"\nLast post details:\n"
            f"  Client: {post_data.get('client', '?')}\n"
            f"  Headline: {d.headline}\n"
            f"  Subtext: {d.subtext}\n"
            f"  CTA: {d.cta}\n"
            f"  Template: {d.template}\n"
            f"  Font: {d.font_headline} ({d.font_headline_weight})\n"
            f"  Colors: bg={d.color_bg}, text={d.color_text}, accent={d.color_accent}\n"
            f"  Has logo: {'yes' if post_data.get('logo_b64') else 'no'}\n"
        )

    return f"""You are John, a creative design assistant for Lectus Creative Engine.
You help Filip create social media posts, learn his design taste, and manage templates.
You speak naturally in whatever language the user speaks (Greek or English).
Keep responses SHORT and conversational — 1-2 sentences. Don't over-explain.

State:
- Has recent taste analysis: {has_analysis}
- Has recent post: {has_post}
{post_context}
Rules:
- If user wants to modify the EXISTING post (translate, change text, adjust colors, etc.), use edit_post.
  "Make the same but in greek" = edit_post with translated text, NOT generate_post.
- If user wants a completely NEW concept/design, use generate_post.
- You can call multiple tools in sequence. E.g. "I love it, make me another one" → save_favorite then generate_post.
- For edit_post, YOU decide the exact field values. Don't ask the user for hex codes — just pick good ones.
- When translating, translate headline, subtext, AND cta. Keep the same tone and meaning.
- Only change the MINIMUM fields needed for edit_post.
- If the user just wants to chat or says hi, respond naturally without calling any tools.
- COLOR CHANGES: When user says "replace X with Y", change ALL fields containing that color (color_bg, color_accent, color_text, color_subtext). Use DISTINCT, clearly different hex values — never subtle variations.
  Reference: Red=#DC2626, Orange=#EA580C, Amber=#D97706, Yellow=#EAB308, Lime=#65A30D, Green=#16A34A, Emerald=#059669, Teal=#0D9488, Cyan=#0891B2, Sky=#0284C7, Blue=#2563EB, Indigo=#4F46E5, Violet=#7C3AED, Purple=#9333EA, Fuchsia=#C026D3, Pink=#DB2777, Rose=#E11D48, White=#FFFFFF, Black=#000000, Gray=#6B7280, Beige=#F5F0E8, Navy=#1E3A5F, Burgundy=#800020, Gold=#FFD700, Coral=#FF6B6B, Turquoise=#40E0D0, Peach=#FFCBA4, Lavender=#E6E6FA, Mint=#98FB98, Cream=#FFFDD0, Charcoal=#36454F
- STOCK PHOTOS: When user asks to "use stock photos", "use real photos", "no AI images", "use actual photos", or anything similar → set image_source="stock" on generate_post. This is CRITICAL. Default is "auto".
- INSPIRATION POSTS: When user sends an image and says "make a post like this", "similar to this one", "based on this" → set use_last_inspiration=true on generate_post. This makes the template replicate the layout of THAT specific image. Do NOT set this for normal posts."""


# ── Photo handler (taste ingestion + logo detection) ─────
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photos — Claude decides if it's a logo/brand asset or inspiration."""
    msg = update.message
    if not msg or not msg.photo:
        return

    photo = msg.photo[-1]
    caption = msg.caption or ""
    user_id = msg.from_user.id if msg.from_user else 0
    history_text = _get_history_text_simple(user_id)

    status_msg = await msg.reply_text("🔍 Looking at this...")

    try:
        file = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        await file.download_to_drive(str(tmp_path))

        # Step 1: Claude classifies what this image is
        image_intent = await _classify_image(tmp_path, caption, history_text)
        logger.info(f"[photo] classified as: {image_intent}")

        if image_intent.get("type") == "logo":
            # ── LOGO / BRAND ASSET ──
            await _handle_logo_upload(msg, status_msg, tmp_path, caption, image_intent, user_id)
            return

        # ── INSPIRATION (default) ──
        analysis = await analyze_inspiration(tmp_path, context=caption)

        client = image_intent.get("client") or _extract_client_from_caption(caption)

        brain.store(
            topic="taste_reference",
            source="telegram",
            content=json.dumps(analysis, ensure_ascii=False),
            client=client or "ALL",
            summary=(
                f"{analysis.get('composition', {}).get('template_match', '?')} layout, "
                f"{analysis.get('feeling', {}).get('mood', '?')} mood"
            ),
            tags=["telegram", analysis.get("composition", {}).get("template_match", "unknown")],
        )

        await status_msg.delete()

        analysis_text = format_analysis_for_telegram(analysis)
        if client:
            analysis_text = f"🏷️ Tagged for: <b>{client}</b>\n\n" + analysis_text

        # Send in chunks if too long for Telegram (4096 char limit)
        # Save analysis BEFORE trying to send — so "make like this" works even if send fails
        _last_analysis_by_user[user_id] = analysis
        _add_to_history(user_id, "user", f"[sent inspiration photo] {caption}")
        _add_to_history(user_id, "assistant", analysis_text[:300])

        # Try HTML first, fall back to plain text if Telegram rejects the formatting
        reply = None
        try:
            if len(analysis_text) <= 4000:
                reply = await msg.reply_text(analysis_text, parse_mode="HTML")
            else:
                chunks = [analysis_text[i:i + 4000] for i in range(0, len(analysis_text), 4000)]
                for chunk in chunks:
                    reply = await msg.reply_text(chunk, parse_mode="HTML")
        except Exception as html_err:
            logger.warning(f"HTML parse failed, sending as plain text: {html_err}")
            # Strip HTML tags and send as plain text
            import re as _re
            plain = _re.sub(r'<[^>]+>', '', analysis_text)
            if len(plain) <= 4000:
                reply = await msg.reply_text(plain)
            else:
                chunks = [plain[i:i + 4000] for i in range(0, len(plain), 4000)]
                for chunk in chunks:
                    reply = await msg.reply_text(chunk)

        if reply:
            _analysis_by_msg[reply.message_id] = analysis

        # If there's a caption, let Claude decide what to do with it
        if caption:
            logger.info(f"Photo has caption — letting Claude decide: '{caption[:60]}'")
            request_text = (
                f"[User sent an inspiration image which I just analyzed. "
                f"The analysis is now stored as the latest inspiration. "
                f"The user's message with the image was: \"{caption}\"]\n\n"
                f"Decide what to do: if the user is asking you to create/generate/make a post based on this image, "
                f"call generate_post with use_last_inspiration=true. "
                f"If they're just sharing inspiration or adding context, respond naturally."
            )
            _add_to_history(user_id, "user", request_text)
            await _run_tool_use_loop(user_id, msg)
            return

    except Exception as e:
        logger.error(f"Photo analysis failed: {e}")
        try:
            await status_msg.edit_text(f"❌ Analysis failed: {str(e)[:200]}")
        except Exception:
            pass  # Status message may already be deleted


def _get_history_text_simple(user_id: int) -> str:
    """Get a simple text summary of history for non-API uses (photo handler etc)."""
    hist = _chat_history.get(user_id, [])
    lines = []
    for msg in hist[-10:]:
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content[:200]
        elif isinstance(content, list):
            # Extract text from content blocks
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            text = " ".join(texts)[:200]
        else:
            text = str(content)[:200]
        prefix = "User" if msg["role"] == "user" else "Bot"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


async def _classify_image(image_path: Path, caption: str, history: str) -> dict:
    """Use Claude Vision to classify if an image is a logo or inspiration."""
    img_bytes = image_path.read_bytes()
    import base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    suffix = image_path.suffix.lower()
    media = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    try:
        response = _ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=(
                "You classify images sent to a design bot. Is this a LOGO/brand asset or INSPIRATION (design reference)?\n\n"
                "Consider the caption and conversation history for context.\n"
                "Return JSON: {\"type\": \"logo\" or \"inspiration\", \"client\": \"ClientName\" or null, \"reason\": \"brief reason\"}\n"
                "Only return \"logo\" if the image is clearly a logo, icon, or brand mark.\n"
                f"Caption: {caption}\nRecent conversation:\n{history}\n\n"
                "Return ONLY JSON."
            ),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": img_b64}},
                    {"type": "text", "text": f"Classify this image. Caption: {caption}"},
                ],
            }],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rstrip("`").strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Image classification failed: {e}")
        return {"type": "inspiration", "client": None}


async def _handle_logo_upload(msg, status_msg, image_path: Path, caption: str, intent: dict, user_id: int):
    """Save a logo/brand asset to the Brain for a specific client."""
    import base64

    client = intent.get("client", "ALL")
    img_bytes = image_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Store logo in Brain
    brain.store(
        topic="brand_logo",
        source="telegram",
        content=json.dumps({
            "image_b64": img_b64,
            "client": client,
            "caption": caption,
            "reason": intent.get("reason", ""),
        }, ensure_ascii=False),
        client=client,
        summary=f"Logo for {client}",
        tags=["logo", "brand_asset", client.lower()],
    )

    await status_msg.delete()
    reply = f"✅ Logo saved for {client}! I'll use it when making posts for {client}."
    await msg.reply_text(reply)
    _add_to_history(user_id, "user", f"[uploaded logo for {client}] {caption}")
    _add_to_history(user_id, "assistant", reply)

    # Clean up
    try:
        image_path.unlink()
    except Exception:
        pass


def _extract_client_from_caption(caption: str) -> str | None:
    if not caption:
        return None
    lower = caption.lower()
    try:
        clients = brain.get_clients(active_only=True)
        for c in clients:
            name = c["name"]
            if name.lower() in lower:
                return name
    except Exception:
        pass
    import re
    match = re.search(r'\bfor\s+(\w+)', caption, re.IGNORECASE)
    if match:
        return match.group(1).title()
    return None


# ── Tool Execution ────────────────────────────────────────

async def _execute_tool(tool_name: str, tool_input: dict, user_id: int, msg) -> str:
    """Execute a tool and return a result string for Claude."""
    logger.info(f"[tool] Executing {tool_name} with {json.dumps(tool_input, ensure_ascii=False)[:200]}")

    if tool_name == "generate_post":
        return await _exec_generate_post(tool_input, user_id, msg)
    elif tool_name == "generate_carousel":
        return await _exec_generate_carousel(tool_input, user_id, msg)
    elif tool_name == "edit_post":
        return await _exec_edit_post(tool_input, user_id, msg)
    elif tool_name == "replace_image":
        return await _exec_replace_image(tool_input, user_id, msg)
    elif tool_name == "save_favorite":
        return await _exec_save_favorite(tool_input, user_id, msg)
    elif tool_name == "process_feedback":
        return await _exec_process_feedback(tool_input, user_id)
    elif tool_name == "get_taste_profile":
        return await _exec_get_taste(user_id, msg)
    elif tool_name == "manage_templates":
        return await _exec_manage_templates(tool_input, msg)
    elif tool_name == "resend_last_post":
        return await _exec_resend(user_id, msg)
    else:
        return f"Unknown tool: {tool_name}"


async def _exec_generate_post(params: dict, user_id: int, msg) -> str:
    """Generate a new post from scratch."""
    client_name = params.get("client", "ALL")
    brief = params.get("brief", "creative post")
    platform = params.get("platform", "linkedin")
    image_source = params.get("image_source", "auto")
    use_last_inspiration = params.get("use_last_inspiration", False)

    # Get the specific inspiration reference if requested
    forced_reference = None
    if use_last_inspiration and user_id in _last_analysis_by_user:
        forced_reference = _last_analysis_by_user[user_id]
        logger.info("Using user's last inspiration image as forced template reference")

    status_msg = await msg.reply_text(
        f"🎨 Generating post for {client_name}...\n"
        f"Brief: {brief}\n\n⏳ This takes 60-120 seconds.",
    )

    async def on_progress(step: str, progress_msg: str):
        try:
            emojis = {
                "research": "🔍", "brain": "🧠", "concept": "💡",
                "copy": "📝", "image": "📸", "decisions": "🎯",
                "template": "📐", "render": "🖨️", "critique": "👁️",
                "fix": "🔧", "brain_write": "💾",
            }
            emoji = emojis.get(step, "⏳")
            await status_msg.edit_text(
                f"🎨 Generating for {client_name}...\n\n{emoji} {progress_msg}",
            )
        except Exception:
            pass

    pipeline_input = PipelineInput(client=client_name, brief=brief, platform=platform)
    prev = _previous_decisions.get(user_id)
    result = await run_pipeline(pipeline_input, brain, on_progress=on_progress, previous_decisions=prev, image_source=image_source, forced_reference=forced_reference)
    result_text = format_result_for_telegram(result, pipeline_input)

    if result.success and result.image_path:
        try:
            img_path = Path(result.image_path)
            # Compress if file is too large for Telegram (max 10MB, aim for < 5MB)
            send_path = img_path
            if img_path.stat().st_size > 5 * 1024 * 1024:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as pil_img:
                    compressed = img_path.with_suffix(".jpg")
                    pil_img.convert("RGB").save(compressed, "JPEG", quality=85)
                    send_path = compressed
                    logger.info(f"Compressed image: {img_path.stat().st_size // 1024}KB → {compressed.stat().st_size // 1024}KB")
            await msg.reply_photo(
                photo=open(send_path, "rb"),
                caption=result_text[:1024],
                parse_mode="HTML",
                read_timeout=60,
                write_timeout=60,
                connect_timeout=30,
            )
            if len(result_text) > 1024:
                await msg.reply_text(result_text[1024:], parse_mode="HTML")
        except Exception as e:
            logger.error(f"Failed to send image: {e}")
            await msg.reply_text(result_text, parse_mode="HTML")

        if result.decisions and result.hero_image:
            _last_post_by_user[user_id] = {
                "decisions": result.decisions,
                "image": result.hero_image,
                "concept": result.concept,
                "client": client_name,
                "template_html": result.template_html,
                "logo_b64": result.logo_b64,
                "rendered_path": result.image_path,
            }
            _save_last_posts()
            if user_id not in _previous_decisions:
                _previous_decisions[user_id] = []
            _previous_decisions[user_id].append({
                "template": result.decisions.template,
                "font": result.decisions.font_headline,
                "color_bg": result.decisions.color_bg,
                "color_text": result.decisions.color_text,
                "color_accent": result.decisions.color_accent,
            })
        return f"Post generated for {client_name}. Image sent to user."
    else:
        await status_msg.edit_text(result_text)
        return f"Post generation failed: {result.error}"


async def _exec_generate_carousel(params: dict, user_id: int, msg) -> str:
    """Generate a carousel of cohesive posts."""
    client_name = params.get("client", "ALL")
    brief = params.get("brief", "creative carousel")
    count = min(params.get("count", 6), 10)
    platform = params.get("platform", "linkedin")

    status_msg = await msg.reply_text(
        f"🎠 Generating carousel of {count} posts for {client_name}...\n"
        f"Theme: {brief}\n\n⏳ This takes {count * 45}-{count * 90} seconds."
    )

    async def on_progress(step: str, progress_msg: str):
        try:
            emojis = {
                "research": "🔍", "brain": "🧠", "concept": "💡",
                "copy": "📝", "image": "📸", "decisions": "🎯",
                "template": "📐", "render": "🖨️", "critique": "👁️",
                "fix": "🔧", "brain_write": "💾",
                "carousel": "🎠",
            }
            emoji = emojis.get(step, "⏳")
            await status_msg.edit_text(
                f"🎠 Carousel for {client_name} ({count} posts)...\n\n{emoji} {progress_msg}",
            )
        except Exception:
            pass

    pipeline_input = PipelineInput(client=client_name, brief=brief, platform=platform)
    results = await run_carousel(pipeline_input, brain, count=count, on_progress=on_progress)
    successful = [r for r in results if r.success and r.image_path]

    if not successful:
        await status_msg.edit_text("❌ Carousel generation failed.")
        return "Carousel generation failed. No posts created."

    media = []
    for i, r in enumerate(successful):
        img_path = Path(r.image_path)
        caption = ""
        if i == 0:
            caption = f"🎠 Carousel for {client_name} ({len(successful)} posts)\n\nTheme: {brief}"
        media.append(InputMediaPhoto(
            media=open(img_path, "rb"),
            caption=caption[:1024] if caption else None,
        ))

    try:
        await msg.reply_media_group(media=media)
    except Exception as e:
        logger.error(f"Failed to send album: {e}")
        for r in successful:
            try:
                await msg.reply_photo(photo=open(Path(r.image_path), "rb"), read_timeout=60, write_timeout=60)
            except Exception:
                pass

    # Send details summary
    summary_lines = [f"🎠 <b>Carousel for {client_name}</b> — {len(successful)}/{count} posts\n"]
    for i, r in enumerate(successful, 1):
        if r.concept and r.decisions:
            summary_lines.append(
                f"<b>Slide {i}:</b> {r.decisions.headline}\n"
                f"   <i>{r.concept.object[:60]}</i>"
            )
    summary = "\n".join(summary_lines)
    try:
        await msg.reply_text(summary, parse_mode="HTML")
    except Exception:
        await msg.reply_text(summary[:4000])

    if successful:
        last = successful[-1]
        if last.decisions and last.hero_image:
            _last_post_by_user[user_id] = {
                "decisions": last.decisions,
                "image": last.hero_image,
                "concept": last.concept,
                "client": client_name,
                "template_html": getattr(last, 'template_html', ''),
                "logo_b64": getattr(last, 'logo_b64', None),
                "rendered_path": last.image_path,
            }
            _save_last_posts()

    return f"Carousel of {len(successful)} posts generated for {client_name}. Images sent."


async def _exec_edit_post(changes: dict, user_id: int, msg) -> str:
    """Edit the last generated post with the given changes."""
    if user_id not in _last_post_by_user:
        return "No recent post to edit. Generate one first."

    post_data = _last_post_by_user[user_id]
    decisions = post_data["decisions"]
    image = post_data["image"]
    client_name = post_data["client"]

    status_msg = await msg.reply_text("✏️ Editing your post...")

    try:
        # Handle special actions
        logo_b64 = post_data.get("logo_b64")
        needs_template_regen = False

        if changes.pop("add_logo", None):
            from pipeline.orchestrator import _get_client_logo
            fetched_logo = _get_client_logo(brain, client_name)
            if fetched_logo:
                logo_b64 = fetched_logo
                needs_template_regen = True
                logger.info(f"[edit] Adding logo for {client_name}")
            else:
                await status_msg.edit_text(f"⚠️ No logo found for {client_name}. Send me the logo first!")
                return f"No logo found for {client_name}."

        if changes.pop("remove_logo", None):
            logo_b64 = None
            needs_template_regen = True
            logger.info(f"[edit] Removing logo")

        # Apply changes to decisions
        new_decisions = replace(decisions, **{
            k: v for k, v in changes.items()
            if hasattr(decisions, k)
        })

        await status_msg.edit_text("✏️ Re-rendering with your changes...")

        template_html = post_data.get("template_html")

        # Regenerate template if needed
        if needs_template_regen or ("template" in changes and changes["template"] != decisions.template):
            reason = "logo change" if needs_template_regen else f"template change"
            logger.info(f"[edit] Regenerating HTML: {reason}")
            template_html = await generate_dynamic_template(new_decisions, brain, has_logo=logo_b64 is not None)

        render_result = await render_post(new_decisions, image, client_name, dynamic_html=template_html, logo_b64=logo_b64, original_decisions=decisions)

        # Build change summary
        change_descriptions = []
        if needs_template_regen and logo_b64:
            change_descriptions.append(f"  • Added {client_name} logo")
        elif needs_template_regen and not logo_b64:
            change_descriptions.append("  • Removed logo")
        for key, val in changes.items():
            if hasattr(decisions, key):
                old_val = getattr(decisions, key)
                if old_val != val:
                    change_descriptions.append(f"  • {key}: {old_val} → {val}")

        changes_text = "\n".join(change_descriptions) if change_descriptions else "  (minor adjustments)"
        result_text = f"✅ Post edited for {client_name}\n\n✏️ Changes:\n{changes_text}"

        img_path = Path(render_result.final_image_path)
        await msg.reply_photo(photo=open(img_path, "rb"), caption=result_text[:1024], read_timeout=60, write_timeout=60)

        _last_post_by_user[user_id] = {
            "decisions": new_decisions,
            "image": image,
            "concept": post_data.get("concept"),
            "client": client_name,
            "template_html": template_html,
            "logo_b64": logo_b64,
            "rendered_path": render_result.final_image_path,
        }
        _save_last_posts()

        return f"Post edited. Changes: {changes_text}"

    except Exception as e:
        logger.error(f"Edit failed: {e}")
        await status_msg.edit_text(f"❌ Edit failed: {str(e)[:200]}")
        return f"Edit failed: {str(e)[:100]}"


async def _exec_replace_image(params: dict, user_id: int, msg) -> str:
    """Replace the hero image in the last post."""
    if user_id not in _last_post_by_user:
        return "No recent post to reimage. Generate one first."

    post_data = _last_post_by_user[user_id]
    decisions = post_data["decisions"]
    concept = post_data.get("concept")
    client_name = post_data["client"]
    template_html = post_data.get("template_html")
    logo_b64 = post_data.get("logo_b64")

    if not concept:
        return "No concept stored from the last post — generate a new one."

    status_msg = await msg.reply_text("🔄 Finding a new image for this concept...")

    try:
        image_source = params.get("image_source", "auto")
        pipeline_input = PipelineInput(client=client_name, brief=concept.object)
        brain_ctx = await brain_read(pipeline_input, brain)
        new_image = await generate_image(concept, brain_ctx, image_source=image_source)
        await status_msg.edit_text(f"🖼️ Got new image ({new_image.model_used}), re-rendering...")

        render_result = await render_post(
            decisions, new_image, client_name,
            dynamic_html=template_html, logo_b64=logo_b64,
        )

        result_text = (
            f"✅ Image replaced for {client_name}\n\n"
            f"🖼️ New image: {new_image.model_used}\n"
            f"📐 Layout unchanged"
        )

        img_path = Path(render_result.final_image_path)
        await msg.reply_photo(photo=open(img_path, "rb"), caption=result_text[:1024], read_timeout=60, write_timeout=60)

        _last_post_by_user[user_id] = {
            "decisions": decisions,
            "image": new_image,
            "concept": concept,
            "client": client_name,
            "template_html": template_html,
            "logo_b64": logo_b64,
            "rendered_path": render_result.final_image_path,
        }
        _save_last_posts()

        return f"Image replaced ({new_image.model_used}). Same layout."

    except Exception as e:
        logger.error(f"Reimage failed: {e}")
        await status_msg.edit_text(f"❌ Image replacement failed: {str(e)[:200]}")
        return f"Reimage failed: {str(e)[:100]}"


async def _exec_save_favorite(params: dict, user_id: int, msg) -> str:
    """Save the current post as a favorite."""
    if user_id not in _last_post_by_user:
        return "No recent post to save."

    last = _last_post_by_user[user_id]
    decisions = last.get("decisions")
    if not decisions:
        return "No decisions to save."

    modifications = params.get("modifications")
    client_name = params.get("client") or last.get("client", "ALL")
    concept_summary = decisions.headline if hasattr(decisions, "headline") else ""

    try:
        await save_liked_template(
            brain=brain,
            decisions=decisions,
            template_html="",
            concept_summary=concept_summary,
            client=client_name,
            modifications=modifications,
        )

        if modifications and client_name != "ALL":
            await save_client_preference(brain, client_name, modifications)

        if modifications:
            changes = ", ".join(f"{k}: {v}" for k, v in modifications.items())
            reply = f"❤️ Saved with changes ({changes}) for {client_name}!"
        else:
            reply = f"❤️ Saved to favorites for {client_name}! I'll use this style more often."

        await msg.reply_text(reply)
        return reply

    except Exception as e:
        logger.error(f"Save favorite failed: {e}")
        await msg.reply_text("❤️ Noted!")
        return "Saved (with minor error in details)."


async def _exec_process_feedback(params: dict, user_id: int) -> str:
    """Process feedback on taste analysis."""
    original_analysis = _last_analysis_by_user.get(user_id)
    if not original_analysis:
        return "No recent analysis to give feedback on."

    try:
        feedback = await parse_feedback(
            user_message=params.get("feedback_text", ""),
            original_analysis=original_analysis,
            brain=brain,
        )
        response_text = format_feedback_response(feedback)
        if user_id in _last_analysis_by_user:
            del _last_analysis_by_user[user_id]
        return response_text
    except Exception as e:
        return f"Feedback processing failed: {str(e)[:100]}"


async def _exec_get_taste(user_id: int, msg) -> str:
    """Show taste profile."""
    try:
        summary = await get_taste_summary(brain)
        await msg.reply_text(summary, parse_mode="HTML")
        return "Taste profile sent."
    except Exception as e:
        return f"Failed to get taste: {str(e)[:100]}"


async def _exec_manage_templates(params: dict, msg) -> str:
    """Show or rebuild templates."""
    action = params.get("action", "show")

    if action == "rebuild":
        status_msg = await msg.reply_text("🔄 Rebuilding templates...\nThis takes 30-60 seconds.")
        try:
            results = await build_templates(brain)
            summary = format_templates_summary(results)
            await status_msg.edit_text(summary, parse_mode="HTML")
            return "Templates rebuilt."
        except ValueError as e:
            await status_msg.edit_text(f"⚠️ {str(e)}")
            return str(e)
        except Exception as e:
            await status_msg.edit_text(f"❌ Failed: {str(e)[:200]}")
            return f"Template rebuild failed: {str(e)[:100]}"
    else:
        templates_dir = Path(config.TEMPLATES_DIR)
        if not templates_dir.exists() or not list(templates_dir.glob("*.html")):
            await msg.reply_text("📐 No templates yet. Feed 10+ images first, then say 'rebuild templates'.")
            return "No templates available."
        template_files = sorted(templates_dir.glob("*.html"))
        lines = ["📐 Current Templates\n"]
        for t in template_files:
            lines.append(f"  • {t.stem}")
        await msg.reply_text("\n".join(lines))
        return f"Showed {len(template_files)} templates."


async def _exec_resend(user_id: int, msg) -> str:
    """Re-send the last generated post."""
    if user_id not in _last_post_by_user:
        await msg.reply_text("🤷 No recent post. Make one first!")
        return "No recent post to resend."

    post_data = _last_post_by_user[user_id]
    rendered = post_data.get("rendered_path")
    if rendered and Path(rendered).exists():
        await msg.reply_photo(
            photo=open(Path(rendered), "rb"),
            caption=f"📎 Last post for {post_data.get('client', '?')}",
            read_timeout=60, write_timeout=60,
        )
        return "Last post re-sent."
    else:
        await msg.reply_text("⚠️ The file is no longer available. Try generating a new one.")
        return "File not available."


# ── Tool-Use Loop (shared by text_handler and photo_handler) ──

async def _run_tool_use_loop(user_id: int, msg) -> None:
    """Run the Claude tool-use loop. Expects history to already contain the user message."""
    system = _build_system_prompt(user_id)
    messages = _get_history_for_api(user_id)
    if not messages:
        return

    try:
        response = await asyncio.to_thread(
            _ai_client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        while response.stop_reason == "tool_use":
            tool_results = []
            assistant_content = response.content

            for block in response.content:
                if block.type == "tool_use":
                    logger.info(f"[tool-use] Claude called {block.name}")
                    result = await _execute_tool(block.name, block.input, user_id, msg)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            assistant_content_dicts = []
            for block in assistant_content:
                if block.type == "text":
                    assistant_content_dicts.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content_dicts.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            _add_to_history(user_id, "assistant", assistant_content_dicts)
            _add_to_history(user_id, "user", tool_results)

            messages = _get_history_for_api(user_id)
            response = await asyncio.to_thread(
                _ai_client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

        final_text = ""
        final_content_dicts = []
        for block in response.content:
            if block.type == "text":
                final_text += block.text
                final_content_dicts.append({"type": "text", "text": block.text})

        if final_content_dicts:
            _add_to_history(user_id, "assistant", final_content_dicts)

        if final_text.strip():
            try:
                await msg.reply_text(final_text.strip())
            except Exception as e:
                logger.warning(f"Failed to send text response: {e}")

    except Exception as e:
        logger.error(f"Tool-use handler failed: {e}", exc_info=True)
        try:
            await msg.reply_text(
                "🎨 Hey! Something went wrong. Try again or say 'make a post for [client]'."
            )
        except Exception:
            pass


# ── Main Text Handler ─────────────────────────────────────

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all text messages using Claude tool-use. Claude decides what to do."""
    msg = update.message
    if not msg or not msg.text:
        return

    text = msg.text.strip()
    user_id = msg.from_user.id if msg.from_user else 0
    logger.info(f"[text] Received from {user_id}: {text[:80]}")

    _add_to_history(user_id, "user", text)
    await _run_tool_use_loop(user_id, msg)


# ── /start command ────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎨 Lectus Creative Engine\n\n"
        "Just talk to me naturally:\n"
        "📸 Send photos → I learn your design taste\n"
        "💬 Reply → confirm, correct, or direct\n"
        "🎯 'Make a post for Somamed about X' → I generate it\n"
        "🎨 'What's my taste?' → I show you\n"
        "📐 'Rebuild templates' → I regenerate from your taste"
    )


# ── Drive watcher scheduler ──────────────────────────────
async def _drive_poll_job():
    try:
        count = await drive_watcher.poll_once()
        if count > 0:
            logger.info(f"Drive poll: processed {count} new image(s)")
            drive_watcher.save_seen_ids()
    except Exception as e:
        logger.error(f"Drive poll error: {e}")


# ── Main ──────────────────────────────────────────────────
def main():
    logger.info("Starting Lectus Creative Engine...")

    from telegram.request import HTTPXRequest
    request = HTTPXRequest(
        read_timeout=120,
        write_timeout=120,
        connect_timeout=30,
        pool_timeout=30,
    )
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).request(request).build()

    # Error handler to catch unhandled exceptions
    async def error_handler(update, context):
        logger.error(f"Unhandled exception: {context.error}", exc_info=context.error)

    # Only /start — everything else is natural language
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(error_handler)

    # Drive watcher
    drive_watcher.bot = app.bot
    drive_watcher.load_seen_ids()

    if config.DRIVE_INSPIRATION_FOLDER_ID:
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            _drive_poll_job,
            "interval",
            seconds=config.DRIVE_WATCH_INTERVAL,
            id="drive_poll",
            name="Drive inspiration folder poll",
        )
        scheduler.start()
        logger.info(f"Drive watcher active — polling every {config.DRIVE_WATCH_INTERVAL}s")
    else:
        logger.info("No DRIVE_INSPIRATION_FOLDER_ID — Drive watcher disabled")

    # Restore templates from Big Brain (survive container restarts)
    restored = load_templates_from_brain(brain)
    if restored > 0:
        logger.info(f"Restored {restored} templates from Big Brain")
    else:
        logger.info("No templates in Brain yet — say 'rebuild templates' to generate them")

    # Restore last post data from disk (survive container restarts)
    _load_last_posts()

    logger.info("Bot ready. Polling for Telegram updates...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
