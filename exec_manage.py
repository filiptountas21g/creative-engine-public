"""Management tool handlers.

Covers the 7 tools that manage templates, taste profile, client rules, and resend:
- save_favorite
- delete_template
- process_feedback
- get_taste_profile
- manage_templates
- save_client_rule
- resend_last_post
"""

import json
import logging
from pathlib import Path

import config
from taste.feedback import parse_feedback, format_feedback_response
from taste.memory import get_taste_summary
from taste.template_builder import build_templates, format_templates_summary
from pipeline.steps.dynamic_template import (
    save_liked_template, save_client_preference, save_template_to_drive,
)
from state import (
    _last_analysis_by_user,
    _last_post_by_user,
)
from bot_helpers import (
    brain,
    _compress_for_send,
)

logger = logging.getLogger(__name__)


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
    template_html = last.get("template_html", "")

    # Collect image description from concept + image result
    image_description = ""
    concept_obj = last.get("concept")
    image_obj = last.get("image")
    if concept_obj and hasattr(concept_obj, "object"):
        image_description = concept_obj.object
    if image_obj and hasattr(image_obj, "prompt_used") and image_obj.prompt_used:
        image_description = image_obj.prompt_used  # more specific than concept

    try:
        await save_liked_template(
            brain=brain,
            decisions=decisions,
            template_html=template_html,
            concept_summary=concept_summary,
            client=client_name,
            modifications=modifications,
            image_description=image_description,
        )

        # Also save the actual HTML to Google Drive for reuse
        if template_html:
            drive_id = await save_template_to_drive(client_name, template_html, decisions)
            if drive_id:
                logger.info(f"Template HTML saved to Drive for {client_name}")

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


async def _exec_delete_template(params: dict, user_id: int, msg) -> str:
    """Delete liked template(s) from memory, or blacklist a style the user hates."""
    client_filter = params.get("client")
    delete_all = params.get("delete_all", False)

    try:
        if delete_all:
            brain._execute("DELETE FROM brain_entries WHERE topic = ?", ["liked_template"])
            reply = "🗑️ Deleted ALL saved templates. Starting fresh."
            await msg.reply_text(reply)
            return reply

        # Delete by client only
        if client_filter and user_id not in _last_post_by_user:
            brain._execute(
                "DELETE FROM brain_entries WHERE topic = ? AND client = ?",
                ["liked_template", client_filter],
            )
            reply = f"🗑️ Deleted all saved templates for {client_filter}."
            await msg.reply_text(reply)
            return reply

        # Match current post against saved templates
        last = _last_post_by_user.get(user_id)
        if not last or not last.get("decisions"):
            reply = "No recent post to match. Tell me which client's templates to delete, or say 'delete all templates'."
            await msg.reply_text(reply)
            return reply

        decisions = last["decisions"]
        post_client = client_filter or last.get("client", "ALL")
        logger.info(f"[delete] Current post: font={decisions.font_headline}, style={decisions.template}, client={post_client}")

        # Try to find and delete a saved template that matches
        liked = brain.query(topic="liked_template", limit=50)
        deleted = 0
        for entry in liked:
            try:
                data = json.loads(entry["content"])
                matches_font = data.get("font_headline") == decisions.font_headline
                matches_style = data.get("template_style") == decisions.template
                matches_client = (not client_filter) or entry.get("client") == client_filter

                if matches_font and matches_style and matches_client:
                    entry_id = entry.get("id")
                    if entry_id:
                        brain._execute("DELETE FROM brain_entries WHERE id = ?", [int(entry_id)])
                        deleted += 1
                        logger.info(f"Deleted liked template id={entry_id}: {data.get('template_style')} + {data.get('font_headline')}")
            except (json.JSONDecodeError, KeyError):
                continue

        if deleted > 0:
            reply = f"🗑️ Deleted {deleted} saved template(s) matching this style ({decisions.template} + {decisions.font_headline}). Won't use it again."
        else:
            # This design was never saved as liked — blacklist it so the bot avoids it
            blacklist_data = {
                "template_style": decisions.template,
                "font_headline": decisions.font_headline,
                "color_bg": decisions.color_bg,
                "color_accent": decisions.color_accent,
                "reason": "User disliked this style",
            }
            brain.store(
                topic="disliked_template",
                source="user_feedback",
                content=json.dumps(blacklist_data),
                client=post_client,
                summary=f"Disliked: {decisions.template} + {decisions.font_headline}",
                tags=["disliked", decisions.template, decisions.font_headline],
            )
            logger.info(f"[delete] No matching liked template — blacklisted: {decisions.template} + {decisions.font_headline}")
            reply = f"🚫 This style ({decisions.template} + {decisions.font_headline}) wasn't in your saved favorites, but I've blacklisted it — I won't use this combo again."

        await msg.reply_text(reply)
        return reply

    except Exception as e:
        logger.error(f"Delete template failed: {e}")
        return f"Failed to delete: {str(e)[:100]}"


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


async def _exec_save_client_rule(params: dict, user_id: int, msg) -> str:
    """Save a permanent design rule for a client to the Brain."""
    client = params.get("client", "")
    rule = params.get("rule", "")
    if not client or not rule:
        return "Need a client name and a rule."

    preferences = {"rules": [rule]}
    if params.get("avoid_colors"):
        preferences["avoid_colors"] = params["avoid_colors"]
    if params.get("prefer_colors"):
        preferences["prefer_colors"] = params["prefer_colors"]
    if params.get("prefer_fonts"):
        preferences["prefer_fonts"] = params["prefer_fonts"]
    if params.get("avoid_fonts"):
        preferences["avoid_fonts"] = params["avoid_fonts"]

    await save_client_preference(brain, client, preferences)
    logger.info(f"[rule] Saved client rule for {client}: {rule}")
    return f"Rule saved for {client}: {rule}. This will apply to all future posts."


async def _exec_resend(user_id: int, msg) -> str:
    """Re-send the last generated post."""
    if user_id not in _last_post_by_user:
        await msg.reply_text("🤷 No recent post. Make one first!")
        return "No recent post to resend."

    post_data = _last_post_by_user[user_id]
    rendered = post_data.get("rendered_path")
    if rendered and Path(rendered).exists():
        send_path = _compress_for_send(Path(rendered))
        with open(send_path, "rb") as photo_fh:
            await msg.reply_photo(
                photo=photo_fh,
                caption=f"📎 Last post for {post_data.get('client', '?')}",
                read_timeout=120, write_timeout=120,
            )
        return "Last post re-sent."
    else:
        await msg.reply_text("⚠️ The file is no longer available. Try generating a new one.")
        return "File not available."
