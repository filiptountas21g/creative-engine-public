"""
Lectus Creative Engine — Telegram Bot

No commands. Just talk naturally.
Send photos → AI learns your taste.
Ask for anything → Claude routes it.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
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
from pipeline.orchestrator import run_pipeline, format_result_for_telegram
from pipeline.steps.render import render as render_post
from pipeline.steps.dynamic_template import save_liked_template, save_client_preference, get_client_preferences
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

# Conversation history per user (last N messages for context)
MAX_HISTORY = 15
_chat_history: dict[int, list[dict]] = {}

# Track last pipeline result per user for edits
_last_post_by_user: dict[int, dict] = {}  # {decisions, image, client, result}

# Track previous decisions per user for variety
_previous_decisions: dict[int, list[dict]] = {}


def _add_to_history(user_id: int, role: str, content: str):
    """Add a message to user's conversation history."""
    if user_id not in _chat_history:
        _chat_history[user_id] = []
    _chat_history[user_id].append({"role": role, "content": content})
    # Keep only last N messages
    if len(_chat_history[user_id]) > MAX_HISTORY:
        _chat_history[user_id] = _chat_history[user_id][-MAX_HISTORY:]


def _get_history_text(user_id: int) -> str:
    """Get conversation history formatted for Claude context."""
    history = _chat_history.get(user_id, [])
    if not history:
        return ""
    lines = []
    for msg in history:
        prefix = "You" if msg["role"] == "user" else "Bot"
        # Truncate long messages to save tokens
        text = msg["content"][:300]
        if len(msg["content"]) > 300:
            text += "..."
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


# ── Intent Router ─────────────────────────────────────────

ROUTER_PROMPT = """You classify user messages for a creative design bot. The user feeds inspiration images and generates social media posts.

Intents:
1. "feedback" — user is responding to a taste analysis (confirming, correcting, directing). Examples: "correct", "σωστά", "remove the lime", "not this", "i like the colors but not the font", "more editorial".
2. "taste" — user asks about their taste profile or what the system has learned. Examples: "what's my taste", "show me my preferences", "τι γούστο", "what have you learned about my style".
3. "make" — user wants to generate a completely NEW post from scratch. Examples: "make a post for Somamed", "generate something for LMW", "create a linkedin post", "φτιάξε ένα post", "make a different one with a new concept".
4. "edit" — user wants to TWEAK the last generated post (move things, change colors, change font, resize, switch template). The image stays the same, only the layout/styling changes. Examples: "move the text inside the picture", "make the headline bigger", "change the background to white", "use a different template", "swap the font", "put the text on the right side".
5. "like" — user approves/loves the last generated post. May ALSO ask for a new one in the same message. Examples: "I love this", "this is great", "perfect", "i really like it", "αυτό μου αρέσει", "i love it can you make me another one". If the message ONLY expresses approval with no request for a new post, return "like". If it ALSO asks for a new post, return "like_and_make".
6. "templates" — user asks about templates or wants to rebuild them. Examples: "show templates", "rebuild templates", "regenerate templates", "what templates do I have".
7. "chat" — greeting, question, or anything else. Examples: "hey", "how does this work", "γεια".

Has recent analysis: {has_analysis}
Has recent post: {has_post}

Recent conversation:
{history}

Return ONLY the intent word. Nothing else."""


def _route_intent(text: str, has_analysis: bool, user_id: int = 0) -> str:
    """Quick Sonnet call to route intent."""
    history = _get_history_text(user_id) or "(no previous messages)"
    has_post = user_id in _last_post_by_user
    try:
        response = _ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            system=ROUTER_PROMPT.format(
                has_analysis=has_analysis,
                has_post=has_post,
                history=history,
            ),
            messages=[{"role": "user", "content": text}],
        )
        intent = response.content[0].text.strip().lower().replace(" ", "_").split()[0]
        if intent in ("feedback", "taste", "make", "edit", "like", "like_and_make", "templates", "chat"):
            return intent
        return "feedback" if has_analysis else "chat"
    except Exception:
        return "feedback" if has_analysis else "chat"


# ── Photo handler (taste ingestion) ──────────────────────
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photos — analyze with Claude Vision."""
    msg = update.message
    if not msg or not msg.photo:
        return

    photo = msg.photo[-1]
    caption = msg.caption or ""

    status_msg = await msg.reply_text("🔍 Analyzing this image...")

    try:
        file = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        await file.download_to_drive(str(tmp_path))

        analysis = await analyze_inspiration(tmp_path, context=caption)

        client = _extract_client_from_caption(caption)

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
        reply = None
        if len(analysis_text) <= 4000:
            reply = await msg.reply_text(analysis_text, parse_mode="HTML")
        else:
            # Split at double newlines
            chunks = []
            remaining = analysis_text
            while remaining:
                if len(remaining) <= 4000:
                    chunks.append(remaining)
                    break
                split_at = remaining.rfind("\n\n", 0, 4000)
                if split_at == -1:
                    split_at = remaining.rfind("\n", 0, 4000)
                if split_at == -1:
                    split_at = 4000
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at:].strip()

            for chunk in chunks:
                reply = await msg.reply_text(chunk, parse_mode="HTML")

        if reply:
            _analysis_by_msg[reply.message_id] = analysis
        _last_analysis_by_user[msg.from_user.id] = analysis
        _add_to_history(msg.from_user.id, "user", "[sent inspiration photo]")
        _add_to_history(msg.from_user.id, "assistant", f"[analyzed image: {analysis.get('feeling', {}).get('mood', '?')} mood, {analysis.get('composition', {}).get('template_match', '?')} layout]")

    except Exception as e:
        logger.error(f"Photo analysis failed: {e}")
        try:
            await msg.reply_text(f"❌ Analysis failed: {str(e)[:200]}")
        except Exception:
            pass
    finally:
        try:
            tmp_path.unlink()
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


# ── Text handler (Claude routes everything) ──────────────
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg or not msg.text:
        return

    text = msg.text.strip()
    user_id = msg.from_user.id if msg.from_user else 0

    # Check if replying to an analysis
    original_analysis = None
    if msg.reply_to_message and msg.reply_to_message.message_id in _analysis_by_msg:
        original_analysis = _analysis_by_msg[msg.reply_to_message.message_id]
    elif user_id in _last_analysis_by_user:
        original_analysis = _last_analysis_by_user[user_id]

    has_analysis = original_analysis is not None

    # Track user message in history
    _add_to_history(user_id, "user", text)

    # Route intent
    intent = await asyncio.to_thread(_route_intent, text, has_analysis, user_id)
    logger.info(f"[router] intent={intent} for: {text[:80]}")

    # ── FEEDBACK ─────────────────────────────────────────
    if intent == "feedback" and original_analysis:
        status_msg = await msg.reply_text("📝 Processing your feedback...")
        try:
            feedback = await parse_feedback(
                user_message=text,
                original_analysis=original_analysis,
                brain=brain,
            )
            response_text = format_feedback_response(feedback)
            await status_msg.edit_text(response_text)
            _add_to_history(user_id, "assistant", response_text)
            if user_id in _last_analysis_by_user:
                del _last_analysis_by_user[user_id]
        except Exception as e:
            logger.error(f"Feedback failed: {e}")
            await status_msg.edit_text(f"❌ Failed: {str(e)[:200]}")
        return

    # ── TASTE ────────────────────────────────────────────
    if intent == "taste":
        try:
            summary = await get_taste_summary(brain)
            await msg.reply_text(summary, parse_mode="HTML")
            _add_to_history(user_id, "assistant", summary[:300])
        except Exception as e:
            logger.error(f"Taste summary failed: {e}")
            await msg.reply_text(f"❌ Failed: {str(e)[:200]}")
        return

    # ── MAKE ─────────────────────────────────────────────
    if intent == "make":
        await _handle_make(msg, text, user_id)
        return

    # ── LIKE / LIKE_AND_MAKE ──────────────────────────────
    if intent in ("like", "like_and_make"):
        if user_id in _last_post_by_user:
            last = _last_post_by_user[user_id]
            try:
                # Use Sonnet to extract any modifications and client context
                modifications, client_for_pref = await _extract_like_details(text, last)

                concept_summary = ""
                if hasattr(last.get("decisions"), "headline"):
                    concept_summary = last["decisions"].headline

                client_name = client_for_pref or last.get("client", "ALL")

                await save_liked_template(
                    brain=brain,
                    decisions=last["decisions"],
                    template_html="",
                    concept_summary=concept_summary,
                    client=client_name,
                    modifications=modifications,
                )

                # If there are client-specific preferences, save them too
                if modifications and client_name != "ALL":
                    await save_client_preference(brain, client_name, modifications)

                if modifications:
                    changes = ", ".join(f"{k}: {v}" for k, v in modifications.items())
                    await msg.reply_text(f"❤️ Saved with your changes ({changes}) for {client_name}!")
                else:
                    await msg.reply_text(f"❤️ Saved to favorites for {client_name}! I'll use this style more often.")
                _add_to_history(user_id, "assistant", f"Saved to favorites for {client_name}.")
            except Exception as e:
                logger.error(f"Failed to save liked template: {e}")
                await msg.reply_text("❤️ Noted!")

        if intent == "like_and_make":
            await _handle_make(msg, text, user_id)
        return

    # ── EDIT ──────────────────────────────────────────────
    if intent == "edit":
        if user_id in _last_post_by_user:
            await _handle_edit(msg, text, user_id)
        else:
            await msg.reply_text("🤷 I don't have a recent post to edit. Make one first!")
            _add_to_history(user_id, "assistant", "No recent post to edit.")
        return

    # ── TEMPLATES ────────────────────────────────────────
    if intent == "templates":
        await _handle_templates(msg, text)
        return

    # ── CHAT (default) ───────────────────────────────────
    bot_reply = (
        "🎨 Lectus Creative Engine\n\n"
        "📸 Στείλε μου φωτογραφίες → μαθαίνω το γούστο σου\n"
        "💬 Πες μου αν συμφωνείς → επιβεβαιώνω τις προτιμήσεις\n"
        "🎯 'Φτιάξε post για Somamed' → δημιουργώ\n"
        "🎨 'Τι γούστο έχω;' → σου δείχνω\n"
        "📐 'Rebuild templates' → ξαναφτιάχνω templates"
    )
    await msg.reply_text(bot_reply)
    _add_to_history(user_id, "assistant", bot_reply)


async def _extract_like_details(text: str, last_post: dict) -> tuple[dict | None, str | None]:
    """Use Sonnet to extract modifications and client from a 'like' message.

    Returns (modifications_dict, client_name) — both can be None.
    Examples:
      "I love it" → (None, None)
      "I love it but use the orange of lmw" → ({"color_accent": "#FF6B00"}, "LMW")
      "save this for georgoulis" → (None, "Georgoulis")
    """
    decisions = last_post.get("decisions")
    if not decisions:
        return None, None

    try:
        response = _ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=(
                "The user liked a generated post and may want to save it with modifications.\n"
                "Extract any design changes they want AND which client this is for.\n\n"
                f"Current post details:\n"
                f"  Font: {decisions.font_headline}\n"
                f"  Colors: bg={decisions.color_bg}, text={decisions.color_text}, accent={decisions.color_accent}\n"
                f"  Template: {decisions.template}\n"
                f"  Client: {last_post.get('client', 'ALL')}\n\n"
                "Return JSON:\n"
                '{"modifications": {"color_accent": "#hex", ...} or null, "client": "ClientName" or null}\n\n'
                "Only include fields that the user explicitly wants changed.\n"
                "If they mention a color by name (orange, blue, red), convert to a reasonable hex.\n"
                "If they mention a client name, extract it. Otherwise null.\n"
                "Return ONLY JSON."
            ),
            messages=[{"role": "user", "content": text}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        data = json.loads(raw)
        mods = data.get("modifications")
        client = data.get("client")
        return mods, client
    except Exception as e:
        logger.warning(f"Could not extract like details: {e}")
        return None, None


async def _handle_make(msg, text: str, user_id: int = 0) -> None:
    """Parse natural language make request and run pipeline."""
    # Get conversation history for context
    history = _get_history_text(user_id)

    # Use Sonnet to extract client + brief from natural text (with conversation context)
    try:
        response = _ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=(
                "Extract the client name and brief from a post generation request. "
                "You have access to the recent conversation history — if the user references "
                "a previous post (e.g. 'make a different one', 'try again but more minimal', "
                "'same client but different approach'), use the history to understand what they mean.\n\n"
                f"Recent conversation:\n{history}\n\n"
                "Return JSON: {\"client\": \"name\", \"brief\": \"the full brief including any context from history\", \"platform\": \"linkedin/instagram/facebook\"}\n"
                "If the user asks for a variation, include in the brief what was done before AND what they want different.\n"
                "If no platform specified, default to linkedin. If no client specified, use \"ALL\"."
            ),
            messages=[{"role": "user", "content": text}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        params = json.loads(raw)
    except Exception:
        params = {"client": "ALL", "brief": text, "platform": "linkedin"}

    client_name = params.get("client", "ALL")
    brief = params.get("brief", text)
    platform = params.get("platform", "linkedin")

    status_msg = await msg.reply_text(
        f"🎨 Generating post for {client_name}...\n"
        f"Brief: {brief}\n\n"
        f"⏳ This takes 60-120 seconds.",
    )

    async def on_progress(step: str, progress_msg: str):
        try:
            emojis = {
                "research": "🔍", "brain": "🧠", "concept": "💡",
                "copy": "📝", "image": "🖼️", "decisions": "🎯",
                "template": "📐", "render": "🖨️", "brain_write": "💾",
            }
            emoji = emojis.get(step, "⏳")
            await status_msg.edit_text(
                f"🎨 Generating for {client_name}...\n\n{emoji} {progress_msg}",
            )
        except Exception:
            pass

    pipeline_input = PipelineInput(
        client=client_name,
        brief=brief,
        platform=platform,
    )

    prev = _previous_decisions.get(user_id)
    result = await run_pipeline(pipeline_input, brain, on_progress=on_progress, previous_decisions=prev)
    result_text = format_result_for_telegram(result, pipeline_input)

    if result.success and result.image_path:
        try:
            img_path = Path(result.image_path)
            await msg.reply_photo(
                photo=open(img_path, "rb"),
                caption=result_text[:1024],
                parse_mode="HTML",
            )
            if len(result_text) > 1024:
                await msg.reply_text(result_text[1024:], parse_mode="HTML")
        except Exception as e:
            logger.error(f"Failed to send image: {e}")
            await msg.reply_text(result_text, parse_mode="HTML")
        # Store for edits + track for variety
        if result.decisions and result.hero_image:
            _last_post_by_user[user_id] = {
                "decisions": result.decisions,
                "image": result.hero_image,
                "client": client_name,
            }
            if user_id not in _previous_decisions:
                _previous_decisions[user_id] = []
            _previous_decisions[user_id].append({
                "template": result.decisions.template,
                "font": result.decisions.font_headline,
                "color_bg": result.decisions.color_bg,
                "color_text": result.decisions.color_text,
                "color_accent": result.decisions.color_accent,
            })
        # Track the result in conversation history
        _add_to_history(user_id, "assistant", result_text[:500])
    else:
        # Error messages may contain angle brackets from logs — don't use HTML parse
        await status_msg.edit_text(result_text)
        _add_to_history(user_id, "assistant", result_text[:300])


EDIT_SYSTEM_PROMPT = """You edit the layout/styling of a social media post. You receive the current design decisions and a user request to change something.

Your job: return ONLY the fields that need to change as a JSON object. Leave out anything that stays the same.

Available fields you can change:
- "template": one of "object-hero", "text-dominant", "split", "full-bleed"
- "headline": the headline text
- "subtext": the supporting text
- "cta": call to action text
- "font_headline": Google Font name (e.g. "Inter", "Space Grotesk", "Playfair Display")
- "font_headline_weight": number (400, 500, 600, 700, 800, 900)
- "font_headline_size": pixels (40-120)
- "font_headline_tracking": CSS letter-spacing (e.g. "-0.02em", "0.05em")
- "font_headline_case": "uppercase", "lowercase", "none"
- "color_bg": hex color
- "color_text": hex color
- "color_accent": hex color
- "color_subtext": hex color
- "headline_margin_x": pixels from left edge (0-200)
- "headline_margin_y": pixels from top/bottom (0-200)
- "headline_max_width": CSS value (e.g. "75%", "90%", "50%")
- "image_padding": pixels around image (0-200)

Common requests and what to change:
- "move text inside the picture" → template: "full-bleed" (text overlays on image)
- "make headline bigger" → font_headline_size: increase by 15-20px
- "change background to white" → color_bg: "#FFFFFF"
- "more minimal" → increase image_padding, reduce font_headline_size
- "put text on the right" → template: "split" with adjustments
- "text over the image" → template: "full-bleed"

Return ONLY valid JSON with the changes. No explanation."""


async def _handle_edit(msg, text: str, user_id: int) -> None:
    """Edit the last generated post by modifying decisions and re-rendering."""
    post_data = _last_post_by_user[user_id]
    decisions = post_data["decisions"]
    image = post_data["image"]
    client_name = post_data["client"]

    status_msg = await msg.reply_text("✏️ Editing your post...")

    # Build current state for Claude
    current_state = {
        "template": decisions.template,
        "headline": decisions.headline,
        "subtext": decisions.subtext,
        "cta": decisions.cta,
        "font_headline": decisions.font_headline,
        "font_headline_weight": decisions.font_headline_weight,
        "font_headline_size": decisions.font_headline_size,
        "font_headline_tracking": decisions.font_headline_tracking,
        "font_headline_case": decisions.font_headline_case,
        "color_bg": decisions.color_bg,
        "color_text": decisions.color_text,
        "color_accent": decisions.color_accent,
        "color_subtext": decisions.color_subtext,
        "headline_margin_x": decisions.headline_margin_x,
        "headline_margin_y": decisions.headline_margin_y,
        "headline_max_width": decisions.headline_max_width,
        "image_padding": decisions.image_padding,
    }

    try:
        # Ask Claude what to change
        response = await asyncio.to_thread(
            _ai_client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=EDIT_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Current design:\n{json.dumps(current_state, indent=2)}\n\nUser request: {text}",
            }],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        changes = json.loads(raw)
        logger.info(f"[edit] Changes: {changes}")

        # Apply changes to a copy of decisions
        from dataclasses import replace
        new_decisions = replace(decisions, **{
            k: v for k, v in changes.items()
            if hasattr(decisions, k)
        })

        await status_msg.edit_text("✏️ Re-rendering with your changes...")

        # Re-render with the same image but new decisions
        render_result = await render_post(new_decisions, image, client_name)

        # Build summary of what changed
        change_descriptions = []
        for key, val in changes.items():
            if hasattr(decisions, key):
                old_val = getattr(decisions, key)
                if old_val != val:
                    change_descriptions.append(f"  • {key}: {old_val} → {val}")

        changes_text = "\n".join(change_descriptions) if change_descriptions else "  (minor adjustments)"
        result_text = f"✅ Post edited for {client_name}\n\n✏️ Changes:\n{changes_text}"

        # Send the edited image
        img_path = Path(render_result.final_image_path)
        await msg.reply_photo(
            photo=open(img_path, "rb"),
            caption=result_text[:1024],
        )

        # Update stored post for further edits
        _last_post_by_user[user_id] = {
            "decisions": new_decisions,
            "image": image,
            "client": client_name,
        }
        _add_to_history(user_id, "assistant", result_text[:300])

    except Exception as e:
        logger.error(f"Edit failed: {e}")
        await status_msg.edit_text(f"❌ Edit failed: {str(e)[:200]}")
        _add_to_history(user_id, "assistant", f"Edit failed: {str(e)[:100]}")


async def _handle_templates(msg, text: str) -> None:
    """Handle template requests — show or rebuild."""
    lower = text.lower()
    rebuild_words = ("rebuild", "regenerate", "ξαναφτιάξε", "ξαναχτισε", "rebuild")

    if any(w in lower for w in rebuild_words):
        status_msg = await msg.reply_text(
            "🔄 Rebuilding templates from your taste data...\n"
            "This takes about 30-60 seconds."
        )
        try:
            results = await build_templates(brain)
            summary = format_templates_summary(results)
            await status_msg.edit_text(summary, parse_mode="HTML")
        except ValueError as e:
            await status_msg.edit_text(f"⚠️ {str(e)}")
        except Exception as e:
            logger.error(f"Template rebuild failed: {e}")
            await status_msg.edit_text(f"❌ Failed: {str(e)[:200]}")
    else:
        templates_dir = Path(config.TEMPLATES_DIR)
        if not templates_dir.exists() or not list(templates_dir.glob("*.html")):
            await msg.reply_text(
                "📐 No templates yet. Feed 10+ inspiration images first, "
                "then say 'rebuild templates'."
            )
            return
        template_files = sorted(templates_dir.glob("*.html"))
        lines = ["📐 Current Templates\n"]
        for t in template_files:
            lines.append(f"  • {t.stem}")
        await msg.reply_text("\n".join(lines))


# ── /start command (keep this one) ───────────────────────
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

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Only /start — everything else is natural language
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

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

    logger.info("Bot ready. Polling for Telegram updates...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
