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
from taste.template_builder import build_templates, format_templates_summary
from pipeline.types import PipelineInput
from pipeline.orchestrator import run_pipeline, format_result_for_telegram

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


# ── Intent Router ─────────────────────────────────────────

ROUTER_PROMPT = """You classify user messages for a creative design bot. The user feeds inspiration images and generates social media posts.

Intents:
1. "feedback" — user is responding to a taste analysis (confirming, correcting, directing). Examples: "correct", "σωστά", "love it", "remove the lime", "not this", "i like the colors but not the font", "more editorial".
2. "taste" — user asks about their taste profile or what the system has learned. Examples: "what's my taste", "show me my preferences", "τι γούστο", "what have you learned about my style".
3. "make" — user wants to generate/create a post. Examples: "make a post for Somamed", "generate something for LMW", "create a linkedin post", "φτιάξε ένα post".
4. "templates" — user asks about templates or wants to rebuild them. Examples: "show templates", "rebuild templates", "regenerate templates", "what templates do I have".
5. "chat" — greeting, question, or anything else. Examples: "hey", "how does this work", "γεια".

Has recent analysis: {has_analysis}

Return ONLY the intent word. Nothing else."""


def _route_intent(text: str, has_analysis: bool) -> str:
    """Quick Sonnet call to route intent."""
    try:
        response = _ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            system=ROUTER_PROMPT.format(has_analysis=has_analysis),
            messages=[{"role": "user", "content": text}],
        )
        intent = response.content[0].text.strip().lower().split()[0]
        if intent in ("feedback", "taste", "make", "templates", "chat"):
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

    # Route intent
    intent = await asyncio.to_thread(_route_intent, text, has_analysis)
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
            await status_msg.edit_text(format_feedback_response(feedback))
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
        except Exception as e:
            logger.error(f"Taste summary failed: {e}")
            await msg.reply_text(f"❌ Failed: {str(e)[:200]}")
        return

    # ── MAKE ─────────────────────────────────────────────
    if intent == "make":
        await _handle_make(msg, text)
        return

    # ── TEMPLATES ────────────────────────────────────────
    if intent == "templates":
        await _handle_templates(msg, text)
        return

    # ── CHAT (default) ───────────────────────────────────
    await msg.reply_text(
        "🎨 Lectus Creative Engine\n\n"
        "📸 Στείλε μου φωτογραφίες → μαθαίνω το γούστο σου\n"
        "💬 Πες μου αν συμφωνείς → επιβεβαιώνω τις προτιμήσεις\n"
        "🎯 'Φτιάξε post για Somamed' → δημιουργώ\n"
        "🎨 'Τι γούστο έχω;' → σου δείχνω\n"
        "📐 'Rebuild templates' → ξαναφτιάχνω templates"
    )


async def _handle_make(msg, text: str) -> None:
    """Parse natural language make request and run pipeline."""
    # Use Sonnet to extract client + brief from natural text
    try:
        response = _ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system=(
                "Extract the client name and brief from a post generation request. "
                "Return JSON: {\"client\": \"name\", \"brief\": \"the brief\", \"platform\": \"linkedin/instagram/facebook\"}\n"
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
                "render": "🖨️", "brain_write": "💾",
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

    result = await run_pipeline(pipeline_input, brain, on_progress=on_progress)
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
    else:
        # Error messages may contain angle brackets from logs — don't use HTML parse
        await status_msg.edit_text(result_text)


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

    logger.info("Bot ready. Polling for Telegram updates...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
