"""Edit-post tool executor.

Applies user-requested modifications to the last generated post — text, colors,
fonts, layout, logo, background removal, and freeform HTML/CSS feedback. Runs a
verify-fix loop via Vision to confirm the edit actually landed.

This lives outside bot.py because `_exec_edit_post` is the largest single
function in the tool-use dispatcher (~330 lines). It pulls shared session
state from `state.py` and a handful of helpers + clients from `bot.py`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path

from pipeline.types import CreativeConcept, ImageResult, PipelineInput
from pipeline.orchestrator import _get_client_logo
from pipeline.steps.render import (
    render as render_post,
    diagnose_layout,
    _format_layout_diagnosis,
    _inject_into_template,
)
from pipeline.steps.image_gen import generate_image, remove_background
from pipeline.steps.brain_read import brain_read
from pipeline.steps.dynamic_template import (
    describe_element_for_generation,
    generate_dynamic_template,
    fix_template_from_critique,
)
from pipeline.steps.critique import check_edit_applied

from state import _last_analysis_by_user, _last_post_by_user
from bot_helpers import brain, _compress_for_send, _track_post_by_msg_id, _persist_user_post

logger = logging.getLogger(__name__)


async def _exec_edit_post(changes: dict, user_id: int, msg) -> str:
    """Edit the last generated post with the given changes."""
    if user_id not in _last_post_by_user:
        return "No recent post to edit. Generate one first."

    post_data = _last_post_by_user[user_id]
    decisions = post_data["decisions"]
    image = post_data["image"]
    client_name = post_data["client"]
    canvas_format = post_data.get("canvas_format", "square")

    status_msg = await msg.reply_text("✏️ Editing your post...")

    try:
        # Handle special actions
        logo_b64 = post_data.get("logo_b64")
        needs_template_regen = False

        if changes.pop("add_logo", None):
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

        # Remove background from hero image
        if changes.pop("remove_background", None):
            if image and image.image_path:
                await status_msg.edit_text("✂️ Removing background...")
                new_path = await remove_background(image.image_path)
                if new_path != image.image_path:
                    image = ImageResult(
                        image_path=new_path,
                        image_url=image.image_url,
                        model_used=image.model_used,
                        prompt_used=image.prompt_used,
                    )
                    logger.info(f"[edit] Background removed: {new_path}")
                else:
                    logger.warning(f"[edit] Background removal returned same path — may have failed")
            else:
                logger.warning(f"[edit] No hero image to remove background from")

        # Extract freeform feedback (handled separately via Opus HTML fix)
        user_feedback = changes.pop("feedback", None)
        add_element = changes.pop("add_element", None)

        # Auto-generate feedback for color changes — CSS variables alone don't work
        # when dynamic templates have full-bleed images or hardcoded colors in divs/overlays.
        # Opus needs to actually edit the HTML to make color changes visible.
        color_feedback_parts = []
        if "color_bg" in changes and changes["color_bg"] != decisions.color_bg:
            color_feedback_parts.append(
                f"Change the VISIBLE background color to {changes['color_bg']}. "
                f"If an image covers the background, add a colored overlay or tint. "
                f"Make sure {changes['color_bg']} is clearly visible as the dominant background color."
            )
        if "color_accent" in changes and changes["color_accent"] != decisions.color_accent:
            color_feedback_parts.append(
                f"Change accent/highlight elements to {changes['color_accent']}."
            )
        if "color_text" in changes and changes["color_text"] != decisions.color_text:
            color_feedback_parts.append(
                f"Change text color to {changes['color_text']}."
            )
        if color_feedback_parts:
            color_feedback = " ".join(color_feedback_parts)
            user_feedback = f"{user_feedback} {color_feedback}" if user_feedback else color_feedback
            logger.info(f"[edit] Auto-generated color feedback for Opus: {color_feedback[:100]}")

        # Apply changes to decisions
        new_decisions = replace(decisions, **{
            k: v for k, v in changes.items()
            if hasattr(decisions, k)
        })

        await status_msg.edit_text("✏️ Re-rendering with your changes...")

        template_html = post_data.get("template_html")
        extra_images = post_data.get("extra_images") or []
        reference_image_b64 = post_data.get("reference_image_b64")

        # Also check if the user's latest analysis has the image
        if not reference_image_b64 and user_id in _last_analysis_by_user:
            reference_image_b64 = _last_analysis_by_user[user_id].get("_image_b64")

        # Generate a new image element if requested (e.g. "add the woman from the inspiration")
        new_image_slot = None
        generated_prompt = None
        if add_element:
            await status_msg.edit_text("👁️ Opus is analyzing the reference to write the image prompt...")
            try:
                # Step 1: Opus looks at the reference image and writes a precise generation prompt
                if reference_image_b64:
                    generated_prompt = await describe_element_for_generation(reference_image_b64, add_element)
                    logger.info(f"[edit] Opus wrote prompt from reference: {generated_prompt[:100]}")
                else:
                    generated_prompt = add_element
                    logger.info(f"[edit] No reference image — using raw description: {add_element}")

                await status_msg.edit_text(f"🖼️ Generating: {generated_prompt[:50]}...")

                # Step 2: Generate the image using Opus's prompt
                concept = post_data.get("concept")
                pipeline_input = PipelineInput(client=client_name, brief=generated_prompt)
                brain_ctx = await brain_read(pipeline_input, brain)

                if concept:
                    element_concept = replace(
                        concept,
                        object=generated_prompt,
                        why=f"Visual element: {add_element}",
                        composition_note="Single subject, clean background, suitable for compositing into a design layout",
                        what_to_avoid="text, logos, busy backgrounds, multiple subjects",
                    )
                else:
                    element_concept = CreativeConcept(
                        object=generated_prompt,
                        why=f"Visual element: {add_element}",
                        emotional_direction="professional, clean",
                        background_color="#ffffff",
                        lighting="studio lighting",
                        composition_note="Single subject, clean background",
                        what_to_avoid="text, logos, busy backgrounds",
                    )

                new_element_image = await generate_image(element_concept, brain_ctx, image_source="ai")
                new_image_slot = len(extra_images) + 2  # +2 because IMAGE_1 is the hero
                extra_images = list(extra_images) + [new_element_image]
                logger.info(f"[edit] Generated element image as IMAGE_{new_image_slot}")
            except Exception as e:
                logger.error(f"[edit] Failed to generate element image: {e}")
                await status_msg.edit_text(f"⚠️ Couldn't generate the image, but applying other changes...")

        # Regenerate template if needed
        if needs_template_regen or ("template" in changes and changes["template"] != decisions.template):
            reason = "logo change" if needs_template_regen else f"template change"
            logger.info(f"[edit] Regenerating HTML: {reason}")
            template_html = await generate_dynamic_template(new_decisions, brain, has_logo=logo_b64 is not None)

        # Pick up user screenshot if they sent one (visual edit feedback)
        user_screenshot_b64 = post_data.pop("user_screenshot_b64", None)

        # Apply freeform feedback — Opus directly edits the HTML/CSS (now with reference image)
        if (user_feedback or new_image_slot) and template_html:
            await status_msg.edit_text("✏️ Opus is adjusting the layout...")

            # If we generated a new image, tell Opus about the available slot
            image_slot_info = ""
            if new_image_slot:
                slot_placeholder = "{{IMAGE_" + str(new_image_slot) + "}}"
                desc = generated_prompt or add_element or "new visual element"
                image_slot_info = (
                    f"\n\n⚠️ NEW IMAGE AVAILABLE: A new image has been generated and is ready at placeholder {slot_placeholder}\n"
                    f"Description: {desc[:200]}\n"
                    f"You MUST add an <img> tag using src=\"{slot_placeholder}\" in the HTML where the user wants it.\n"
                    f"Style it with appropriate CSS (position, size, border-radius, object-fit, etc).\n"
                )

            feedback_text = user_feedback or f"Add the new image element ({add_element}) to the design."

            # If user sent a screenshot, tell Opus about it
            screenshot_hint = ""
            if user_screenshot_b64:
                screenshot_hint = (
                    "\n\n📸 USER SCREENSHOT ATTACHED: The user sent a screenshot (possibly cropped) "
                    "showing exactly what they want changed. Look at the USER SCREENSHOT image carefully "
                    "to understand what element they're pointing at.\n"
                )

            critique_text = (
                f"USER FEEDBACK — fix these issues in the HTML/CSS:\n\n"
                f"[CRITICAL] user_feedback: {feedback_text}\n"
                f"  → Fix: Apply the user's requested change to the template HTML/CSS.\n"
                f"{screenshot_hint}"
                f"{image_slot_info}"
            )
            logger.info(f"[edit] Applying feedback via Opus (ref={bool(reference_image_b64)}, screenshot={bool(user_screenshot_b64)}): {feedback_text[:80]}")
            template_html = await fix_template_from_critique(
                template_html, critique_text, new_decisions,
                reference_image_b64=reference_image_b64,
                user_screenshot_b64=user_screenshot_b64,
            )

        # Safety net: if template lost image placeholders but we have images, re-inject them
        _existing_slots = set(re.findall(r'\{\{IMAGE_(\d+)\}\}', template_html or ""))
        _total_images = 1 + len(extra_images) if image else len(extra_images)
        if _total_images > 0 and not _existing_slots:
            logger.warning(f"[edit] Template has NO image placeholders but {_total_images} images exist — re-injecting")
            # Build a simple image grid at the bottom
            _img_tags = []
            for _i in range(1, _total_images + 1):
                _img_tags.append(
                    f'<img src="{{{{IMAGE_{_i}}}}}" style="width:{100//_total_images}%;height:300px;'
                    f'object-fit:cover;object-position:top center;" alt="photo {_i}">'
                )
            _grid_html = (
                '<div style="position:absolute;bottom:80px;left:40px;right:40px;'
                'display:flex;gap:16px;z-index:2;">'
                + "".join(_img_tags)
                + '</div>'
            )
            template_html = template_html.replace("</body>", f"{_grid_html}\n</body>")
            logger.info(f"[edit] Injected {_total_images} image placeholders into template")

        render_result = await render_post(new_decisions, image, client_name, dynamic_html=template_html, logo_b64=logo_b64, original_decisions=decisions, extra_images=extra_images or None, canvas_format=canvas_format)

        # ── Verify-fix loop: check if freeform feedback was actually applied ──
        if user_feedback and template_html:
            max_verify_rounds = 2
            previous_attempts: list[str] = []
            for verify_round in range(1, max_verify_rounds + 1):
                check = await check_edit_applied(render_result.final_image_path, user_feedback)
                applied = check.get("applied", False)
                confidence = check.get("confidence", 5)

                if applied:
                    logger.info(f"[edit] Feedback verified as applied (round {verify_round}, confidence={confidence})")
                    break

                # Not applied but Vision isn't confident — accept rather than risk breaking it
                if not applied and confidence < 5:
                    logger.info(f"[edit] Uncertain (confidence={confidence}) — accepting as-is")
                    break

                fix_instruction = check.get("fix_instruction", "")
                if not fix_instruction:
                    logger.info(f"[edit] Not applied but no fix instruction — accepting")
                    break

                logger.info(f"[edit] Feedback NOT applied (confidence={confidence}, round {verify_round}) — retrying: {fix_instruction[:100]}")
                await status_msg.edit_text(f"🔄 Adjusting... (attempt {verify_round + 1})")

                # DOM inspection — get actual bounding boxes and detected overlaps
                layout_info = ""
                try:
                    _injected_html = _inject_into_template(
                        template_html, new_decisions, image, client_name,
                        logo_b64, decisions, extra_images=extra_images or None,
                        canvas_format=canvas_format,
                    )
                    diag = await diagnose_layout(_injected_html, canvas_format=canvas_format)
                    layout_info = "\n\n" + _format_layout_diagnosis(diag)
                except Exception as _e:
                    logger.warning(f"[edit] Layout diagnosis failed: {_e}")

                # Track attempts so Opus knows what already failed
                previous_attempts.append(fix_instruction)
                attempt_history = ""
                if len(previous_attempts) > 1:
                    attempt_history = (
                        f"\n\n⚠️ PREVIOUS ATTEMPTS FAILED — do NOT repeat them:\n"
                        + "\n".join(f"  Attempt {i+1}: {a[:150]}" for i, a in enumerate(previous_attempts[:-1]))
                        + "\n\nYou must try a DIFFERENT technique this time."
                    )

                retry_critique = (
                    f"The user asked: \"{user_feedback}\"\n"
                    f"This was NOT applied. Vision says: {check.get('what_i_see', '')}\n\n"
                    f"Vision's suggested fix: {fix_instruction}"
                    f"{layout_info}\n\n"
                    f"🔍 Use the LAYOUT DIAGNOSIS above — it has the EXACT bounding boxes and CSS positioning for each element. "
                    f"Base your fix on those concrete numbers, not visual estimation:\n"
                    f"  • If elements use `position: absolute` → change `top:`/`bottom:`/`left:`/`right:` values (margin is IGNORED).\n"
                    f"  • If elements are in flex/grid → use `gap:` or adjust alignment.\n"
                    f"  • If elements are in normal flow → margin/padding works.\n"
                    f"  • If overlap is shown in px, move the lower element down by at least that many px + 10px buffer.{attempt_history}\n\n"
                    f"⚠️ SAFETY: Do NOT remove the headline, subtext, CTA, hero image, or any core content. "
                    f"Only make the MINIMUM change needed to resolve the overlap. The fix must be SURGICAL."
                )
                template_html = await fix_template_from_critique(
                    template_html, retry_critique, new_decisions,
                    reference_image_b64=reference_image_b64,
                    rendered_image_path=render_result.final_image_path,
                    user_screenshot_b64=user_screenshot_b64,
                )
                render_result = await render_post(
                    new_decisions, image, client_name, dynamic_html=template_html,
                    logo_b64=logo_b64, original_decisions=decisions,
                    extra_images=extra_images or None, canvas_format=canvas_format,
                )

        # Build change summary
        change_descriptions = []
        if add_element:
            change_descriptions.append(f"  • Added element: {add_element[:60]}")
        if user_feedback:
            change_descriptions.append(f"  • Layout adjusted: {user_feedback}")
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

        img_path = _compress_for_send(Path(render_result.final_image_path))
        with open(img_path, "rb") as photo_fh:
            sent_edit_msg = await msg.reply_photo(photo=photo_fh, caption=result_text[:1024], read_timeout=120, write_timeout=120)

        edit_post_data = {
            "decisions": new_decisions,
            "image": image,
            "concept": post_data.get("concept"),
            "client": client_name,
            "template_html": template_html,
            "logo_b64": logo_b64,
            "rendered_path": render_result.final_image_path,
            "extra_images": extra_images,
            "canvas_format": canvas_format,
        }
        _last_post_by_user[user_id] = edit_post_data
        _persist_user_post(user_id)

        # Track by message ID so user can reply to save it later
        if sent_edit_msg:
            _track_post_by_msg_id(sent_edit_msg.message_id, user_id, edit_post_data)

        return f"Post edited. Changes: {changes_text}"

    except Exception as e:
        logger.error(f"Edit failed: {e}")
        await status_msg.edit_text(f"❌ Edit failed: {str(e)[:200]}")
        return f"Edit failed: {str(e)[:100]}"
