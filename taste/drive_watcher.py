"""
Google Drive folder watcher for inspiration images.

Polls a Drive folder every N seconds for new images,
downloads them, runs Claude Vision analysis, and sends
results via Telegram.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

import config
from brain.client import Brain
from brain.drive_client import list_new_images, download_file
from taste.vision import analyze_inspiration, format_analysis_for_telegram

logger = logging.getLogger(__name__)


class DriveWatcher:
    """Watches a Google Drive folder for new inspiration images."""

    def __init__(self, brain: Brain, telegram_bot=None):
        self.brain = brain
        self.bot = telegram_bot
        self.seen_ids: set = set()
        self.folder_id = config.DRIVE_INSPIRATION_FOLDER_ID
        self._analysis_context: dict = {}  # msg_id → analysis for feedback

    async def check_for_new_images(self) -> list[dict]:
        """Check Drive folder for new images. Returns list of new file metadata."""
        if not self.folder_id:
            return []

        try:
            new_files = await list_new_images(self.folder_id, self.seen_ids)
            if new_files:
                logger.info(f"Found {len(new_files)} new image(s) in Drive")
            return new_files
        except Exception as e:
            logger.error(f"Drive watch error: {e}")
            return []

    async def process_new_image(self, file_meta: dict) -> dict | None:
        """Download, analyze, store, and notify about a new image."""
        file_id = file_meta["id"]
        file_name = file_meta.get("name", "unknown.jpg")

        logger.info(f"Processing new Drive image: {file_name}")

        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            await download_file(file_id, tmp_path)

            # Analyze with Claude Vision
            analysis = await analyze_inspiration(tmp_path, context=f"From Drive: {file_name}")

            # Store in Big Brain
            import json
            self.brain.store(
                topic="taste_reference",
                source="drive_watcher",
                content=json.dumps(analysis, ensure_ascii=False),
                summary=(
                    f"Analyzed {file_name}: "
                    f"{analysis.get('composition', {}).get('template_match', '?')} layout, "
                    f"{analysis.get('feeling', {}).get('mood', '?')} mood"
                ),
                tags=["drive", analysis.get("composition", {}).get("template_match", "unknown")],
            )

            # Mark as seen
            self.seen_ids.add(file_id)

            # Send Telegram notification
            if self.bot and config.TELEGRAM_CHAT_ID:
                message_text = (
                    f"📁 <b>New from Drive:</b> {file_name}\n\n"
                    + format_analysis_for_telegram(analysis)
                )
                try:
                    # Send photo + analysis
                    msg = await self.bot.send_photo(
                        chat_id=config.TELEGRAM_CHAT_ID,
                        photo=open(tmp_path, "rb"),
                        caption=message_text[:1024],  # Telegram caption limit
                        parse_mode="HTML",
                    )
                    # If analysis is longer than caption, send rest as text
                    if len(message_text) > 1024:
                        await self.bot.send_message(
                            chat_id=config.TELEGRAM_CHAT_ID,
                            text=message_text[1024:],
                            parse_mode="HTML",
                            reply_to_message_id=msg.message_id,
                        )
                    # Store for feedback tracking
                    self._analysis_context[msg.message_id] = analysis
                except Exception as e:
                    logger.error(f"Failed to send Telegram notification: {e}")

            logger.info(f"Processed Drive image: {file_name}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to process Drive image {file_name}: {e}")
            return None
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink()
            except Exception:
                pass

    async def poll_once(self) -> int:
        """Run one poll cycle. Returns number of images processed."""
        new_files = await self.check_for_new_images()
        processed = 0

        for file_meta in new_files:
            result = await self.process_new_image(file_meta)
            if result:
                processed += 1
            # Small delay between images to avoid rate limits
            await asyncio.sleep(2)

        return processed

    def load_seen_ids(self):
        """Load previously seen file IDs from Brain agent memory."""
        saved = self.brain.get_memory("taste_engine", "drive_seen_ids")
        if saved and isinstance(saved, list):
            self.seen_ids = set(saved)
            logger.info(f"Loaded {len(self.seen_ids)} seen Drive file IDs")

    def save_seen_ids(self):
        """Save seen file IDs to Brain agent memory."""
        self.brain.set_memory("taste_engine", "drive_seen_ids", list(self.seen_ids))
