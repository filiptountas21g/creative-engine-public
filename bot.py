"""
Lectus Creative Engine — Telegram Bot

Two modes:
  1. TASTE: Send photos → AI breaks them down → stores in Big Brain → you direct
  2. GENERATE: /make command → full pipeline → returns PNG
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

# Track analyses for feedback (message_id → analysis dict)
_analysis_by_msg: dict[int, dict] = {}
# Track which message a user is replying to
_photo_msg_to_analysis: dict[int, dict] = {}


# ── Photo handler (taste ingestion) ──────────────────────
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photos — analyze with Claude Vision and respond."""
    msg = update.message
    if not msg or not msg.photo:
        return

    # Get the highest resolution photo
    photo = msg.photo[-1]
    caption = msg.caption or ""

    # Send "analyzing..." message
    status_msg = await msg.reply_text("🔍 Analyzing this image...")

    try:
        # Download photo to temp file
        file = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        await file.download_to_drive(str(tmp_path))

        # Run Claude Vision analysis
        analysis = await analyze_inspiration(tmp_path, context=caption)

        # Store in Big Brain
        # Check if caption mentions a client
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

        # Delete status message
        await status_msg.delete()

        # Send analysis
        analysis_text = format_analysis_for_telegram(analysis)
        if client:
            analysis_text = f"🏷️ Tagged for: <b>{client}</b>\n\n" + analysis_text

        reply = await msg.reply_text(analysis_text, parse_mode="HTML")

        # Track for feedback
        _analysis_by_msg[reply.message_id] = analysis
        _photo_msg_to_analysis[msg.message_id] = analysis

    except Exception as e:
        logger.error(f"Photo analysis failed: {e}")
        await status_msg.edit_text(f"❌ Analysis failed: {str(e)[:200]}")
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def _extract_client_from_caption(caption: str) -> str | None:
    """Try to extract a client name from the photo caption."""
    if not caption:
        return None

    lower = caption.lower()

    # Patterns: "for Somamed", "this is for Somamed", "Somamed"
    # Check against known clients
    try:
        clients = brain.get_clients(active_only=True)
        for c in clients:
            name = c["name"]
            if name.lower() in lower:
                return name
    except Exception:
        pass

    # Pattern: "for <word>"
    import re
    match = re.search(r'\bfor\s+(\w+)', caption, re.IGNORECASE)
    if match:
        return match.group(1).title()

    return None


# ── Text handler (feedback on analyses) ──────────────────
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text replies to analyses — parse feedback."""
    msg = update.message
    if not msg or not msg.text:
        return

    # Check if this is a reply to an analysis
    if msg.reply_to_message and msg.reply_to_message.message_id in _analysis_by_msg:
        original_analysis = _analysis_by_msg[msg.reply_to_message.message_id]

        status_msg = await msg.reply_text("📝 Processing your feedback...")

        try:
            feedback = await parse_feedback(
                user_message=msg.text,
                original_analysis=original_analysis,
                brain=brain,
            )

            await status_msg.edit_text(format_feedback_response(feedback))

        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            await status_msg.edit_text(f"❌ Failed to process feedback: {str(e)[:200]}")
        return

    # Not a reply to an analysis — ignore or handle other text
    # (Could be a general chat message)


# ── /taste command ────────────────────────────────────────
async def taste_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show taste summary. Usage: /taste [fonts|colors|composition|<client>]"""
    args = context.args
    aspect = None
    client_filter = None

    if args:
        arg = args[0].lower()
        if arg in ("fonts", "font", "typography"):
            aspect = "fonts"
        elif arg in ("colors", "colour", "color", "palette"):
            aspect = "colors"
        elif arg in ("composition", "layout", "layouts"):
            aspect = "composition"
        else:
            # Assume it's a client name
            client_filter = args[0]

    summary = await get_taste_summary(brain, aspect=aspect)

    # If client-specific, add client info
    if client_filter:
        from taste.memory import get_taste_context
        ctx = await get_taste_context(brain, client=client_filter)
        client_data = ctx.get("client_specific", {}).get(client_filter)
        if client_data:
            summary += f"\n\n🏷️ <b>{client_filter}:</b>\n{client_data}"
        else:
            summary += f"\n\nNo specific taste data for {client_filter} yet."

    await update.message.reply_text(summary, parse_mode="HTML")


# ── /make command (generation pipeline) ───────────────────
async def make_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a post. Usage: /make <client> <brief>"""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /make <client> <brief>\n"
            "Example: /make Somamed launching new anti-aging treatment"
        )
        return

    client = context.args[0]
    brief = " ".join(context.args[1:])

    # Check for --template flag
    template_override = None
    if "--template" in brief:
        parts = brief.split("--template")
        brief = parts[0].strip()
        if len(parts) > 1:
            template_override = parts[1].strip().split()[0] if parts[1].strip() else None

    status_msg = await update.message.reply_text(
        f"🎨 Generating post for <b>{client}</b>...\n"
        f"Brief: {brief}\n\n"
        f"⏳ This takes 60-120 seconds. I'll update you on progress.",
        parse_mode="HTML",
    )

    # Progress callback
    async def on_progress(step: str, msg: str):
        try:
            step_emojis = {
                "research": "🔍", "brain": "🧠", "concept": "💡",
                "copy": "📝", "image": "🖼️", "decisions": "🎯",
                "render": "🖨️", "brain_write": "💾",
            }
            emoji = step_emojis.get(step, "⏳")
            await status_msg.edit_text(
                f"🎨 Generating post for <b>{client}</b>...\n\n"
                f"{emoji} {msg}",
                parse_mode="HTML",
            )
        except Exception:
            pass  # Telegram edit rate limits

    # Run pipeline
    pipeline_input = PipelineInput(
        client=client,
        brief=brief,
        platform="linkedin",
        template_override=template_override,
    )

    result = await run_pipeline(pipeline_input, brain, on_progress=on_progress)

    # Send result
    result_text = format_result_for_telegram(result, pipeline_input)

    if result.success and result.image_path:
        try:
            from pathlib import Path
            img_path = Path(result.image_path)
            await update.message.reply_photo(
                photo=open(img_path, "rb"),
                caption=result_text[:1024],
                parse_mode="HTML",
            )
            if len(result_text) > 1024:
                await update.message.reply_text(
                    result_text[1024:], parse_mode="HTML"
                )
        except Exception as e:
            logger.error(f"Failed to send generated image: {e}")
            await update.message.reply_text(result_text, parse_mode="HTML")
    else:
        await status_msg.edit_text(result_text, parse_mode="HTML")


# ── /forget command ───────────────────────────────────────
async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove a taste reference. Usage: /forget <id>"""
    # TODO: Implement deletion by ID
    await update.message.reply_text(
        "🗑️ /forget is not yet implemented. "
        "Use /taste to review your preferences."
    )


# ── /templates command ────────────────────────────────────
async def templates_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current templates."""
    templates_dir = Path(config.TEMPLATES_DIR)
    if not templates_dir.exists() or not list(templates_dir.glob("*.html")):
        await update.message.reply_text(
            "📐 No templates generated yet.\n"
            "Feed 10+ inspiration images first, then use /rebuild-templates."
        )
        return

    template_files = sorted(templates_dir.glob("*.html"))
    lines = ["📐 <b>Current Templates</b>\n"]
    for t in template_files:
        lines.append(f"  • {t.stem}")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


# ── /rebuild-templates command ────────────────────────────
async def rebuild_templates_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Regenerate HTML templates from taste data."""
    status_msg = await update.message.reply_text(
        "🔄 Rebuilding templates from your taste data...\n"
        "This takes about 30-60 seconds (4 templates to generate)."
    )

    try:
        results = await build_templates(brain)
        summary = format_templates_summary(results)
        await status_msg.edit_text(summary, parse_mode="HTML")
    except ValueError as e:
        await status_msg.edit_text(f"⚠️ {str(e)}")
    except Exception as e:
        logger.error(f"Template rebuild failed: {e}")
        await status_msg.edit_text(f"❌ Template rebuild failed: {str(e)[:200]}")


# ── /start command ────────────────────────────────────────
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message."""
    await update.message.reply_text(
        "🎨 <b>Lectus Creative Engine</b>\n\n"
        "I learn your design taste and generate on-brand social media posts.\n\n"
        "<b>Teach me your taste:</b>\n"
        "  📸 Send a photo → I'll break it down\n"
        "  💬 Reply to my analysis → confirm, correct, or direct\n"
        "  📁 Drop images in Drive → I'll process them\n\n"
        "<b>Commands:</b>\n"
        "  /taste — see your taste profile\n"
        "  /taste fonts — typography preferences\n"
        "  /taste colors — color preferences\n"
        "  /make <client> <brief> — generate a post\n"
        "  /templates — see current templates\n"
        "  /rebuild-templates — regenerate templates from taste\n",
        parse_mode="HTML",
    )


# ── Drive watcher scheduler ──────────────────────────────
async def _drive_poll_job():
    """Scheduled job: poll Drive for new images."""
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

    # Build Telegram app
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("taste", taste_command))
    app.add_handler(CommandHandler("make", make_command))
    app.add_handler(CommandHandler("forget", forget_command))
    app.add_handler(CommandHandler("templates", templates_command))
    app.add_handler(CommandHandler("rebuild_templates", rebuild_templates_command))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Set up Drive watcher with bot reference
    drive_watcher.bot = app.bot
    drive_watcher.load_seen_ids()

    # Schedule Drive polling
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
        logger.info("No DRIVE_INSPIRATION_FOLDER_ID set — Drive watcher disabled")

    logger.info("Bot ready. Polling for Telegram updates...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
