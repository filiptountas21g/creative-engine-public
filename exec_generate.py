"""Post-generation tool handlers.

Covers the 4 tools that create or restore a rendered post:
- generate_post
- generate_carousel
- replace_image
- restore_image

Imports from bot.py happen at module top; they work because bot.py defines
all referenced names before its `from exec_generate import ...` line executes.
See exec_edit.py for the same pattern.
"""

import logging
from dataclasses import replace
from pathlib import Path

from telegram import InputMediaPhoto

from pipeline.types import PipelineInput
from pipeline.orchestrator import run_pipeline, run_carousel, format_result_for_telegram
from pipeline.steps.render import render as render_post
from pipeline.steps.image_gen import generate_image
from pipeline.steps.brain_read import brain_read
from state import (
    _last_analysis_by_user,
    _last_post_by_user,
    _image_vault,
    _previous_decisions,
)
from bot_helpers import (
    brain,
    _compress_for_send,
    _track_post_by_msg_id,
    _persist_user_post,
    _remember,
    _vault_save_images,
    _vault_get,
)

logger = logging.getLogger(__name__)


async def _exec_generate_post(params: dict, user_id: int, msg) -> str:
    """Generate a new post from scratch."""
    client_name = params.get("client", "ALL")
    brief = params.get("brief", "creative post")
    platform = params.get("platform", "linkedin")
    image_source = params.get("image_source", "auto")
    canvas_format = params.get("format", "square")
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
                "decompose": "🔬", "template": "📐", "render": "🖨️",
                "critique": "👁️", "fix": "🔧", "brain_write": "💾",
            }
            emoji = emojis.get(step, "⏳")
            await status_msg.edit_text(
                f"🎨 Generating for {client_name}...\n\n{emoji} {progress_msg}",
            )
        except Exception:
            pass

    style_overrides = params.get("style_overrides", {})

    pipeline_input = PipelineInput(client=client_name, brief=brief, platform=platform, format=canvas_format)
    prev = _previous_decisions.get(user_id)
    result = await run_pipeline(pipeline_input, brain, on_progress=on_progress, previous_decisions=prev, image_source=image_source, forced_reference=forced_reference)
    # Orchestrator may have auto-detected landscape from reference — read it back
    canvas_format = getattr(pipeline_input, "format", canvas_format)

    # Apply style overrides — user specified colors/fonts that override the AI's choices
    if style_overrides and result.success and result.decisions:
        overrides = {k: v for k, v in style_overrides.items() if hasattr(result.decisions, k)}
        if overrides:
            logger.info(f"Applying style overrides: {overrides}")
            result.decisions = replace(result.decisions, **overrides)
            # Re-render with overridden decisions
            try:
                await status_msg.edit_text(f"🎨 Applying your style preferences...")
                render_result = await render_post(
                    result.decisions, result.hero_image, client_name,
                    dynamic_html=result.template_html, logo_b64=result.logo_b64,
                    extra_images=result.extra_images,
                    canvas_format=canvas_format,
                )
                result.image_path = render_result.final_image_path
            except Exception as e:
                logger.error(f"Re-render with overrides failed: {e}")

    result_text = format_result_for_telegram(result, pipeline_input)

    if result.success and result.image_path:
        try:
            img_path = Path(result.image_path)
            send_path = _compress_for_send(img_path)
            with open(send_path, "rb") as photo_fh:
                sent_msg = await msg.reply_photo(
                    photo=photo_fh,
                    caption=result_text[:1024],
                    parse_mode="HTML",
                    read_timeout=120,
                    write_timeout=120,
                    connect_timeout=30,
                )
            if len(result_text) > 1024:
                await msg.reply_text(result_text[1024:], parse_mode="HTML")
        except Exception as e:
            sent_msg = None
            logger.error(f"Failed to send image: {e}")
            await msg.reply_text(result_text, parse_mode="HTML")

        if result.decisions and result.hero_image:
            # Store reference image bytes if available (for Opus to see during edits)
            ref_image_b64 = None
            if forced_reference and isinstance(forced_reference, dict):
                ref_image_b64 = forced_reference.get("_image_b64")

            post_data = {
                "decisions": result.decisions,
                "image": result.hero_image,
                "concept": result.concept,
                "client": client_name,
                "template_html": result.template_html,
                "logo_b64": result.logo_b64,
                "rendered_path": result.image_path,
                "extra_images": result.extra_images,
                "canvas_format": canvas_format,
                "reference_elements": result.reference_elements,
                "reference_image_b64": ref_image_b64,
            }
            _last_post_by_user[user_id] = post_data
            _remember(user_id, "post", post_data, label=f"{client_name}: {result.decisions.headline[:40]}")
            _persist_user_post(user_id)

            # Save all images to vault (never lost across edits)
            _vault_save_images(user_id, hero_image=result.hero_image, extra_images=result.extra_images)

            # Track by message ID so user can reply to this post later
            if sent_msg:
                _track_post_by_msg_id(sent_msg.message_id, user_id, post_data)

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
                "decompose": "🔬", "template": "📐", "render": "🖨️",
                "critique": "👁️", "fix": "🔧", "brain_write": "💾",
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
        send_path = _compress_for_send(Path(r.image_path))
        caption = ""
        if i == 0:
            caption = f"🎠 Carousel for {client_name} ({len(successful)} posts)\n\nTheme: {brief}"
        with open(send_path, "rb") as f:
            photo_bytes = f.read()
        media.append(InputMediaPhoto(
            media=photo_bytes,
            caption=caption[:1024] if caption else None,
        ))

    try:
        await msg.reply_media_group(media=media)
    except Exception as e:
        logger.error(f"Failed to send album: {e}")
        for r in successful:
            try:
                fallback_path = _compress_for_send(Path(r.image_path))
                with open(fallback_path, "rb") as f:
                    await msg.reply_photo(photo=f, read_timeout=120, write_timeout=120)
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
                "extra_images": getattr(last, 'extra_images', None),
                "canvas_format": "square",  # carousels are always square
            }
            _persist_user_post(user_id)

    return f"Carousel of {len(successful)} posts generated for {client_name}. Images sent."


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
    canvas_format = post_data.get("canvas_format", "square")

    if not concept:
        return "No concept stored from the last post — generate a new one."

    status_msg = await msg.reply_text("🔄 Finding a new image for this concept...")

    try:
        image_source = params.get("image_source", "auto")
        background_style = params.get("background_style")
        user_description = params.get("description")
        pipeline_input = PipelineInput(client=client_name, brief=user_description or concept.object)
        brain_ctx = await brain_read(pipeline_input, brain)

        if background_style:
            # User wants a custom AI-generated background — build a minimal concept from the style description
            bg_concept = replace(
                concept,
                object=background_style,
                why="Background visual for the post",
                composition_note="Abstract, no text, no people, suitable as a full-bleed background",
                what_to_avoid="text, logos, faces, complex objects",
            )
            await status_msg.edit_text(f"🎨 Generating AI background: {background_style[:60]}...")
            new_image = await generate_image(bg_concept, brain_ctx, image_source="ai")
        elif user_description:
            # User described what they want — override the concept
            new_concept = replace(
                concept,
                object=user_description,
                why=f"User requested: {user_description}",
                composition_note=concept.composition_note,
            )
            await status_msg.edit_text(f"🔍 Searching for: {user_description[:60]}...")
            new_image = await generate_image(new_concept, brain_ctx, image_source=image_source)
            logger.info(f"[replace] Used user description: {user_description}")
        else:
            new_image = await generate_image(concept, brain_ctx, image_source=image_source)
        await status_msg.edit_text(f"🖼️ Got new image ({new_image.model_used}), re-rendering...")

        extra_images = post_data.get("extra_images")
        render_result = await render_post(
            decisions, new_image, client_name,
            dynamic_html=template_html, logo_b64=logo_b64,
            extra_images=extra_images, canvas_format=canvas_format,
        )

        if background_style:
            result_text = (
                f"✅ Background replaced for {client_name}\n\n"
                f"🎨 Style: {background_style[:80]}\n"
                f"🤖 Model: {new_image.model_used}\n"
                f"📐 Layout unchanged"
            )
        else:
            result_text = (
                f"✅ Image replaced for {client_name}\n\n"
                f"🖼️ New image: {new_image.model_used}\n"
                f"📐 Layout unchanged"
            )

        img_path = _compress_for_send(Path(render_result.final_image_path))
        with open(img_path, "rb") as photo_fh:
            await msg.reply_photo(photo=photo_fh, caption=result_text[:1024], read_timeout=120, write_timeout=120)

        _last_post_by_user[user_id] = {
            "decisions": decisions,
            "image": new_image,
            "concept": concept,
            "client": client_name,
            "template_html": template_html,
            "logo_b64": logo_b64,
            "rendered_path": render_result.final_image_path,
            "extra_images": extra_images,
            "canvas_format": canvas_format,
        }
        _persist_user_post(user_id)

        # Save new image to vault
        _vault_save_images(user_id, hero_image=new_image)

        return f"Image replaced ({new_image.model_used}). Same layout."

    except Exception as e:
        logger.error(f"Reimage failed: {e}")
        await status_msg.edit_text(f"❌ Image replacement failed: {str(e)[:200]}")
        return f"Reimage failed: {str(e)[:100]}"


async def _exec_restore_image(params: dict, user_id: int, msg) -> str:
    """Restore a previously used image from the vault and re-render."""
    if user_id not in _last_post_by_user:
        return "No recent post to restore an image to."

    vault = _image_vault.get(user_id, {})
    if not vault:
        return "No images saved in the vault yet. The vault stores images from this session."

    search = params.get("search")
    restored_image = _vault_get(user_id, search)

    if not restored_image:
        # List what's available
        available = [k for k in vault.keys() if not k.startswith("latest_")]
        return f"Couldn't find a matching image. Available: {', '.join(available[:5])}"

    # Check the image file still exists
    if not Path(restored_image.image_path).exists():
        return "The image file was cleaned up. It's no longer available."

    post_data = _last_post_by_user[user_id]
    decisions = post_data["decisions"]
    client_name = post_data["client"]
    template_html = post_data.get("template_html")
    logo_b64 = post_data.get("logo_b64")
    extra_images = post_data.get("extra_images") or []
    canvas_format = post_data.get("canvas_format", "square")

    status_msg = await msg.reply_text("🔄 Restoring image and re-rendering...")

    try:
        render_result = await render_post(
            decisions, restored_image, client_name,
            dynamic_html=template_html, logo_b64=logo_b64,
            extra_images=extra_images or None, canvas_format=canvas_format,
        )

        img_path = _compress_for_send(Path(render_result.final_image_path))
        prompt_hint = (restored_image.prompt_used or "previous image")[:50]
        result_text = f"✅ Restored image: {prompt_hint}"

        await status_msg.delete()
        with open(img_path, "rb") as photo_fh:
            await msg.reply_photo(photo=photo_fh, caption=result_text[:1024], read_timeout=120, write_timeout=120)

        _last_post_by_user[user_id] = {
            "decisions": decisions,
            "image": restored_image,
            "concept": post_data.get("concept"),
            "client": client_name,
            "template_html": template_html,
            "logo_b64": logo_b64,
            "rendered_path": render_result.final_image_path,
            "extra_images": extra_images,
            "canvas_format": canvas_format,
        }
        _persist_user_post(user_id)

        return f"Image restored: {prompt_hint}"

    except Exception as e:
        logger.error(f"Restore image failed: {e}")
        await status_msg.edit_text(f"❌ Restore failed: {str(e)[:200]}")
        return f"Restore failed: {str(e)[:100]}"
