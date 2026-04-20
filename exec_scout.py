"""Scout tool handlers.

Covers the 3 scout-related tools:
- scout_designs  (Phase 1: Perplexity search → preview to user)
- approve_scout  (Phase 2: user-approved items → stored in Brain)
- generate_from_scout  (shortcut: generate a post using a scout item as forced reference)
"""

import asyncio
import io
import logging
from pathlib import Path

import httpx

from pipeline.types import PipelineInput
from pipeline.orchestrator import run_pipeline, format_result_for_telegram
from pipeline.steps.design_scout import (
    scout_search, scout_approve, detect_staleness, extract_single_reference,
)
from state import (
    _last_post_by_user,
    _previous_decisions,
    _pending_scout,
)
from bot_helpers import (
    brain,
    _compress_for_send,
    _persist_user_post,
    _remember,
)

logger = logging.getLogger(__name__)


async def _exec_scout_designs(params: dict, user_id: int, msg) -> str:
    """Phase 1: Search for fresh design inspiration — show preview, wait for approval."""
    client_name = params.get("client", "ALL")
    focus = params.get("focus", "")
    focus_text = f" ({focus})" if focus else ""
    status_msg = await msg.reply_text(f"🔍 Searching for design inspiration{f' for {client_name}' if client_name != 'ALL' else ''}{focus_text}...")

    try:
        staleness = detect_staleness(brain, client=client_name)

        await status_msg.edit_text("🔍 Building targeted search queries...")
        result = await scout_search(brain, client=client_name, staleness=staleness, user_focus=focus)

        items = result.get("items", [])
        citations = result.get("citations", [])

        if not items:
            await status_msg.edit_text("❌ Couldn't find design references. Try again later.")
            return "Scout search found nothing."

        # Store pending results for approval
        _pending_scout[user_id] = result

        await status_msg.delete()

        # Show staleness notice if needed
        if staleness.get("is_stale"):
            patterns = staleness.get("repeated_patterns", {})
            stale_desc = ", ".join(f"{v}" for v in patterns.values())
            await msg.reply_text(f"⚠️ Detected repetition in recent posts: {stale_desc}")

        await msg.reply_text(f"🔍 Found {len(items)} designs — sending previews...")

        # Send each photo by passing URL directly to Telegram (Telegram downloads it)
        # Falls back to downloading bytes ourselves if Telegram rejects the URL
        sent_count = 0

        async def _send_one(item: dict):
            nonlocal sent_count
            idx = item["index"]
            name = item.get("name", "Design")
            description = item.get("description", "")[:180]
            domain = item.get("domain", "")
            page_url = item.get("url", "")
            image_url = item.get("image_url", "")
            thumbnail_url = item.get("thumbnail_url", "")

            caption = f"<b>{idx}. {name}</b>\n{description}\n📍 {domain}"
            if page_url:
                caption += f"\n🔗 {page_url}"
            caption = caption[:1024]

            # Strategy 1: pass URL directly — Telegram downloads it (no Railway egress)
            urls_to_try_direct = [u for u in [image_url, thumbnail_url] if u]
            for url in urls_to_try_direct:
                try:
                    await msg.reply_photo(photo=url, caption=caption, parse_mode="HTML")
                    logger.info(f"Scout image {idx}: sent via URL ({url[:60]})")
                    sent_count += 1
                    return
                except Exception as e:
                    logger.warning(f"Scout image {idx}: Telegram URL send failed ({url[:60]}): {e}")

            # Strategy 2: download bytes ourselves and upload
            for url in urls_to_try_direct:
                try:
                    async with httpx.AsyncClient(timeout=25, follow_redirects=True) as hc:
                        resp = await hc.get(url, headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                            "Referer": "https://www.google.com/",
                        })
                    if resp.status_code == 200 and len(resp.content) > 3000:
                        await msg.reply_photo(photo=io.BytesIO(resp.content), caption=caption, parse_mode="HTML")
                        logger.info(f"Scout image {idx}: sent via bytes ({len(resp.content)} bytes)")
                        sent_count += 1
                        return
                except Exception as e:
                    logger.warning(f"Scout image {idx}: bytes download failed ({url[:60]}): {e}")

            # Strategy 3: text-only fallback
            logger.warning(f"Scout image {idx}: all strategies failed, sending text")
            try:
                await msg.reply_text(f"{idx}. {name} — {page_url or domain}")
            except Exception:
                pass
            sent_count += 1

        # Send all items sequentially (Telegram rate limits media sends)
        for item in items:
            await _send_one(item)
            await asyncio.sleep(0.3)  # Avoid Telegram flood limits

        # Final instruction
        await msg.reply_text(
            f"💬 Sent {sent_count} designs. Which ones should I save to Brain?\n"
            "Reply with numbers (e.g. <b>1, 3, 5</b>), <b>all</b>, or <b>none</b>.\n"
            "💡 Or say <b>'make a post for [client] like 3'</b> to use one directly!\n"
            "The ones you save teach the system what you like — react to as many as you can.",
            parse_mode="HTML",
        )

        return (
            f"Scout found {len(items)} design references. Preview shown to user with links. "
            f"WAITING for user to respond. They can: "
            f"(1) Reply with numbers to save (e.g. '1, 3, 5') → call approve_scout with selected=[1,3,5]. "
            f"(2) Say 'all' → approve_scout with all=true. "
            f"(3) Say 'none' → approve_scout with selected=[]. "
            f"(4) Say 'make a post for [client] like 3' → call generate_post with scout_layout_index=3. "
            f"Items available: {', '.join(str(i['index']) for i in items)}"
        )

    except Exception as e:
        logger.error(f"Design scout failed: {e}", exc_info=True)
        try:
            await status_msg.edit_text(f"❌ Scout failed: {str(e)[:200]}")
        except Exception:
            pass
        return f"Scout failed: {str(e)[:100]}"


async def _exec_approve_scout(params: dict, user_id: int, msg) -> str:
    """Phase 2: User approved specific scout items — process and store them."""
    pending = _pending_scout.get(user_id)
    if not pending:
        return "No pending scout results. Run scout_designs first."

    items = pending.get("items", [])
    approve_all = params.get("all", False)

    if approve_all:
        selected = [item["index"] for item in items]
    else:
        selected = params.get("selected", [])

    feedback = params.get("feedback", "").strip()

    if not selected:
        # Store rejection reason in Brain so future scouts avoid this
        if feedback:
            brain.store(
                topic="taste_rejected",
                source="scout_feedback",
                content=feedback,
                client=pending.get("client", "ALL"),
                summary=f"Scout rejected: {feedback}",
                tags=["scout_rejection", "avoid"],
            )
            logger.info(f"Stored scout rejection feedback: {feedback}")
            await msg.reply_text(f"👍 Noted — I'll avoid {feedback} next time I search.")
        else:
            await msg.reply_text("👍 No designs saved. Scout results cleared.")
        _pending_scout.pop(user_id, None)
        return f"User declined all scout results.{' Reason: ' + feedback if feedback else ''}"

    status_msg = await msg.reply_text(f"⏳ Processing {len(selected)} approved designs...")

    try:
        result = await scout_approve(brain, pending, selected)

        _pending_scout.pop(user_id, None)  # Clear pending

        if not result.get("stored"):
            await status_msg.edit_text("❌ Failed to process designs.")
            return "Scout approval failed."

        await status_msg.delete()

        # Show stored layouts
        layouts_text = result.get("layouts_text", "")
        header = f"✅ Saved {result.get('layouts_found', 0)} layout blueprints to Brain!\nOpus will use these for fresh designs.\n"

        await msg.reply_text(header)

        if layouts_text:
            chunks = [layouts_text[i:i + 4000] for i in range(0, len(layouts_text), 4000)]
            for chunk in chunks[:3]:
                try:
                    await msg.reply_text(chunk)
                except Exception:
                    pass

        return f"Approved and stored {result.get('layouts_found', 0)} layouts in Brain."

    except Exception as e:
        logger.error(f"Scout approval failed: {e}", exc_info=True)
        try:
            await status_msg.edit_text(f"❌ Processing failed: {str(e)[:200]}")
        except Exception:
            pass
        return f"Approval failed: {str(e)[:100]}"


async def _exec_generate_from_scout(params: dict, user_id: int, msg) -> str:
    """Generate a post using a specific scout item as the forced layout reference.
    Downloads the actual design image → Vision analyzes it → Opus replicates the layout."""
    pending = _pending_scout.get(user_id)
    if not pending:
        return "No pending scout results. Run scout_designs first."

    item_index = params.get("scout_item", 1)
    client_name = params.get("client", "ALL")
    brief = params.get("brief", "creative post")
    platform = params.get("platform", "linkedin")
    image_source = params.get("image_source", "auto")

    # Find the item name for display
    item_name = "design"
    for item in pending.get("items", []):
        if item["index"] == item_index:
            item_name = item.get("name", "design")
            break

    status_msg = await msg.reply_text(
        f"🎨 Generating post for {client_name} using scout layout #{item_index} ({item_name})...\n\n"
        f"⏳ Downloading design image + analyzing layout..."
    )

    try:
        # Step 1: Download image + run full Vision analysis (same as inspiration photos)
        reference = await extract_single_reference(pending, item_index)
        if not reference:
            await status_msg.edit_text(f"❌ Couldn't extract reference for item #{item_index}")
            return f"Failed to extract reference for scout item {item_index}."

        has_image = "_image_b64" in reference
        if has_image:
            await status_msg.edit_text(
                f"🎨 Downloaded + analyzed the design ✓\n"
                f"Building post with {item_name} layout for {client_name}..."
            )
        else:
            await status_msg.edit_text(
                f"⚠️ Couldn't download image — using text reference\n"
                f"Building post with {item_name} layout for {client_name}..."
            )

        # Step 2: Also save this item to Brain (user chose it = implicit approval)
        try:
            await scout_approve(brain, pending, [item_index])
        except Exception as e:
            logger.warning(f"Scout auto-approve on generate failed (non-critical): {e}")

        async def on_progress(step: str, progress_msg: str):
            try:
                emojis = {
                    "research": "🔍", "brain": "🧠", "concept": "💡",
                    "copy": "📝", "image": "📸", "decisions": "🎯",
                    "template": "📐", "render": "🖨️", "critique": "👁️",
                    "fix": "🔧", "brain_write": "💾", "scout": "🔍",
                }
                emoji = emojis.get(step, "⏳")
                await status_msg.edit_text(
                    f"🎨 {item_name} layout for {client_name}...\n\n{emoji} {progress_msg}",
                )
            except Exception:
                pass

        pipeline_input = PipelineInput(client=client_name, brief=brief, platform=platform)
        prev = _previous_decisions.get(user_id)

        result = await run_pipeline(
            pipeline_input, brain, on_progress=on_progress,
            previous_decisions=prev, image_source=image_source,
            forced_reference=reference,
        )

        result_text = format_result_for_telegram(result, pipeline_input)

        if result.success and result.image_path:
            try:
                send_path = _compress_for_send(Path(result.image_path))

                caption = f"🎨 Layout: {item_name}\n{result_text[:900]}"
                with open(send_path, "rb") as photo_fh:
                    await msg.reply_photo(
                        photo=photo_fh,
                        caption=caption[:1024],
                        parse_mode="HTML",
                        read_timeout=120, write_timeout=120, connect_timeout=30,
                    )
            except Exception as e:
                logger.error(f"Failed to send image: {e}")
                await msg.reply_text(result_text, parse_mode="HTML")

            if result.decisions and result.hero_image:
                post_data = {
                    "decisions": result.decisions,
                    "image": result.hero_image,
                    "concept": result.concept,
                    "client": client_name,
                    "template_html": result.template_html,
                    "logo_b64": result.logo_b64,
                    "rendered_path": result.image_path,
                    "extra_images": result.extra_images,
                    "canvas_format": "square",  # scout-generated posts are always square
                }
                _last_post_by_user[user_id] = post_data
                _remember(user_id, "post", post_data, label=f"{client_name}: {result.decisions.headline[:40]}")
                _persist_user_post(user_id)

                if user_id not in _previous_decisions:
                    _previous_decisions[user_id] = []
                _previous_decisions[user_id].append({
                    "template": result.decisions.template,
                    "font": result.decisions.font_headline,
                    "color_bg": result.decisions.color_bg,
                    "color_text": result.decisions.color_text,
                    "color_accent": result.decisions.color_accent,
                })

            return f"Post generated for {client_name} using scout layout '{item_name}'. Image sent."
        else:
            await status_msg.edit_text(result_text)
            return f"Post generation failed: {result.error}"

    except Exception as e:
        logger.error(f"Generate from scout failed: {e}", exc_info=True)
        try:
            await status_msg.edit_text(f"❌ Generation failed: {str(e)[:200]}")
        except Exception:
            pass
        return f"Generation failed: {str(e)[:100]}"
