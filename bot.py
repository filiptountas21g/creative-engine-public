"""
Lectus Creative Engine — Telegram Bot

No commands. Just talk naturally.
Send photos → AI learns your taste.
Ask for anything → Claude decides what to do using tool-use.
"""

import asyncio
import base64
import io
import json
import logging
import re
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import anthropic
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from PIL import Image as PILImage
from telegram import Update, InputMediaPhoto
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

import config
from brain.client import Brain
from taste.vision import analyze_inspiration, format_analysis_for_telegram
from taste.feedback import parse_feedback, format_feedback_response
from taste.memory import get_taste_summary
from taste.drive_watcher import DriveWatcher
from taste.template_builder import build_templates, format_templates_summary, load_templates_from_brain
from pipeline.types import CreativeConcept, CreativeDecisions, ImageResult, PipelineInput
from pipeline.orchestrator import run_pipeline, run_carousel, format_result_for_telegram, _get_client_logo
from pipeline.steps.render import (
    render as render_post,
    diagnose_layout,
    _format_layout_diagnosis,
    _inject_into_template,
)
from pipeline.steps.image_gen import generate_image, remove_background
from pipeline.steps.brain_read import brain_read
from pipeline.steps.dynamic_template import (
    save_liked_template, save_client_preference,
    get_client_preferences, generate_dynamic_template,
    fix_template_from_critique, describe_element_for_generation,
    save_template_to_drive,
)
from pipeline.steps.critique import check_edit_applied
from pipeline.steps.design_scout import (
    scout_search, scout_approve, detect_staleness, extract_single_reference,
)
from tools_schema import TOOLS

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────
# `brain` + session helpers live in bot_helpers.py so exec_*.py modules can
# import them without a circular edge through bot.py.
from bot_helpers import (
    brain,
    _vault_save_images,
    _vault_get,
    _track_post_by_msg_id,
    _persist_user_post,
    _compress_for_send,
    _remember,
)

drive_watcher = DriveWatcher(brain)
_ai_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Per-user session state lives in state.py so other modules can share it.
from state import (
    MAX_HISTORY,
    MAX_TRACKED_POSTS,
    MAX_VAULT_IMAGES,
    MAX_SAVED_REFERENCES,
    _last_analysis_by_user,
    _chat_history,
    _last_post_by_user,
    _posts_by_msg_id,
    _image_vault,
    _previous_decisions,
    _pending_scout,
    _restored_users,
)


def _persist_user_reference(user_id: int):
    """Save the user's last reference analysis + image to Brain (survives restarts).
    Keeps last 5 per user — prunes older ones."""
    if user_id not in _last_analysis_by_user:
        return
    try:
        data = _last_analysis_by_user[user_id].copy()
        # Give it a short label from the analysis for identification
        label = data.get("feeling", {}).get("mood", "") or data.get("composition", {}).get("template_match", "")
        brain.store(
            topic="user_session_reference",
            source=str(user_id),
            content=json.dumps(data, default=str),
            summary=f"{label[:40]}" if label else "reference",
            tags=["session", "reference", str(user_id)],
        )
        # Prune old references beyond MAX_SAVED_REFERENCES
        existing = brain.query(topic="user_session_reference", source=str(user_id), limit=MAX_SAVED_REFERENCES + 10)
        if len(existing) > MAX_SAVED_REFERENCES:
            # Delete the oldest ones (query returns newest first)
            for old in existing[MAX_SAVED_REFERENCES:]:
                try:
                    brain._execute("DELETE FROM brain_entries WHERE id = ?", [old["id"]])
                except Exception as e:
                    logger.debug(f"[persist] Failed to prune old reference id={old.get('id')}: {e}")
            logger.info(f"[persist] Pruned {len(existing) - MAX_SAVED_REFERENCES} old references for user {user_id}")
        logger.info(f"[persist] Saved reference for user {user_id} ({len(data.get('_image_b64', '')) // 1024}KB image)")
    except Exception as e:
        logger.warning(f"[persist] Failed to save reference: {e}")


def _persist_chat_history(user_id: int):
    """Save the user's chat history to Brain (survives restarts).
    Deletes old entry first — only 1 row per user ever exists."""
    hist = _chat_history.get(user_id, [])
    if not hist:
        return
    try:
        # Only persist last 10 messages to keep it small
        # Strip image content blocks (too large) — keep text and tool calls only
        slim_hist = []
        for msg in hist[-10:]:
            if isinstance(msg.get("content"), str):
                slim_hist.append(msg)
            elif isinstance(msg.get("content"), list):
                slim_blocks = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "image":
                        continue  # skip image blocks — too large
                    slim_blocks.append(block)
                if slim_blocks:
                    slim_hist.append({"role": msg["role"], "content": slim_blocks})

        # Delete old entry first — keeps exactly 1 row per user
        brain.delete_by_topic_source("user_session_chat", str(user_id))
        brain.store(
            topic="user_session_chat",
            source=str(user_id),
            content=json.dumps(slim_hist, default=str),
            summary=f"Chat history for user {user_id} ({len(slim_hist)} messages)",
            tags=["session", "chat", str(user_id)],
        )
        logger.info(f"[persist] Saved chat history for user {user_id} ({len(slim_hist)} messages)")
    except Exception as e:
        logger.warning(f"[persist] Failed to save chat history: {e}")


def _restore_user_post(user_id: int):
    """Restore the user's last generated post from Brain after restart.
    Writes image files back to disk so edits work."""
    if user_id in _last_post_by_user:
        return  # Already have a post in memory

    try:
        posts = brain.query(topic="user_session_post", source=str(user_id), limit=1)
        if not posts:
            return

        meta = json.loads(posts[0]["content"])

        # Reconstruct decisions
        decisions = None
        if meta.get("decisions"):
            decisions = CreativeDecisions(**meta["decisions"])

        # Reconstruct concept
        concept = None
        if meta.get("concept"):
            concept = CreativeConcept(**meta["concept"])

        # Restore image files from DB → disk
        img_rows = brain.query(topic="user_session_post_images", source=str(user_id), limit=10)

        hero_image = None
        extra_images = []

        for img_row in img_rows:
            try:
                img_data = json.loads(img_row["content"])
                label = img_data.get("label", "")
                img_path = img_data.get("image_path", "")
                bytes_b64 = img_data.get("bytes_b64", "")

                if bytes_b64 and img_path:
                    # Write image back to disk
                    p = Path(img_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(base64.b64decode(bytes_b64))

                    img_result = ImageResult(
                        image_path=img_path,
                        image_url=img_data.get("image_url", ""),
                        model_used=img_data.get("model_used", ""),
                        prompt_used=img_data.get("prompt_used", ""),
                    )

                    if label == "hero":
                        hero_image = img_result
                    else:
                        extra_images.append(img_result)

                    logger.info(f"[restore] Wrote {label} image to {img_path}")
            except Exception as e:
                logger.warning(f"[restore] Failed to restore image: {e}")

        # Reconstruct hero from metadata if file wasn't in DB
        if not hero_image and meta.get("image"):
            hero_image = ImageResult(**{k: v for k, v in meta["image"].items() if k in ("image_path", "image_url", "model_used", "prompt_used")})

        # Fetch logo from Brain if post had one
        logo_b64 = None
        if meta.get("has_logo", False):
            logo_b64 = _get_client_logo(brain, meta.get("client", ""))

        # Get reference image from restored session reference
        reference_image_b64 = None
        if meta.get("has_reference_image") and user_id in _last_analysis_by_user:
            reference_image_b64 = _last_analysis_by_user[user_id].get("_image_b64")

        _last_post_by_user[user_id] = {
            "decisions": decisions,
            "image": hero_image,
            "concept": concept,
            "client": meta.get("client", ""),
            "template_html": meta.get("template_html", ""),
            "logo_b64": logo_b64,
            "rendered_path": meta.get("rendered_path", ""),
            "extra_images": extra_images or None,
            "canvas_format": meta.get("canvas_format", "square"),
            "reference_elements": meta.get("reference_elements"),
            "reference_image_b64": reference_image_b64,
        }

        n_imgs = (1 if hero_image else 0) + len(extra_images)
        logger.info(f"[restore] Restored post for user {user_id} (client={meta.get('client', '?')}, {n_imgs} images)")
    except Exception as e:
        logger.warning(f"[restore] Failed to restore post: {e}")


def _sanitize_chat_history(hist: list[dict]) -> list[dict]:
    """Clean up restored chat history to prevent API errors.
    Uses ID-based matching: every tool_use must have a tool_result with the same ID
    in the next user message, and every tool_result must reference a tool_use in the
    previous assistant message. Orphans on either side are stripped."""
    if not hist:
        return hist

    # Pass 1: collect all tool_use IDs and tool_result IDs
    tool_use_ids = set()
    tool_result_ids = set()
    for msg in hist:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                tool_use_ids.add(block["id"])
            elif block.get("type") == "tool_result" and block.get("tool_use_id"):
                tool_result_ids.add(block["tool_use_id"])

    # IDs that have both a tool_use and tool_result
    paired_ids = tool_use_ids & tool_result_ids
    orphan_use_ids = tool_use_ids - paired_ids
    orphan_result_ids = tool_result_ids - paired_ids

    for oid in orphan_use_ids:
        logger.warning(f"[sanitize] Dropping orphaned tool_use {oid}")
    for oid in orphan_result_ids:
        logger.warning(f"[sanitize] Dropping orphaned tool_result for {oid}")

    # Pass 2: rebuild history, stripping orphaned blocks
    clean = []
    for msg in hist:
        content = msg.get("content", [])
        if not isinstance(content, list):
            clean.append(msg)
            continue

        filtered_blocks = []
        for block in content:
            if not isinstance(block, dict):
                filtered_blocks.append(block)
                continue
            if block.get("type") == "tool_use" and block.get("id") in orphan_use_ids:
                continue  # drop orphaned tool_use
            if block.get("type") == "tool_result" and block.get("tool_use_id") in orphan_result_ids:
                continue  # drop orphaned tool_result
            filtered_blocks.append(block)

        # If all blocks were stripped, drop the entire message
        if not filtered_blocks:
            continue
        clean.append({**msg, "content": filtered_blocks})

    # Pass 3: ensure alternating user/assistant roles (no two consecutive same-role)
    final = []
    for msg in clean:
        if final and final[-1].get("role") == msg.get("role"):
            # Merge into previous message if same role
            prev_content = final[-1].get("content", [])
            curr_content = msg.get("content", [])
            if isinstance(prev_content, list) and isinstance(curr_content, list):
                final[-1] = {**final[-1], "content": prev_content + curr_content}
            # else just skip the duplicate
        else:
            final.append(msg)

    # Must start with user message
    while final and final[0].get("role") != "user":
        final.pop(0)

    return final


def _restore_user_session(user_id: int):
    """Restore user's reference analysis and chat history from Brain after restart."""
    if user_id in _restored_users:
        return  # Already restored this session
    _restored_users.add(user_id)

    # Restore latest reference analysis
    if user_id not in _last_analysis_by_user:
        try:
            refs = brain.query(topic="user_session_reference", source=str(user_id), limit=1)
            if refs:
                data = json.loads(refs[0]["content"])
                _last_analysis_by_user[user_id] = data
                has_image = bool(data.get("_image_b64"))
                # Count total saved references
                all_refs = brain.query(topic="user_session_reference", source=str(user_id), limit=MAX_SAVED_REFERENCES)
                logger.info(f"[restore] Restored reference for user {user_id} (has_image={has_image}, {len(all_refs)} saved total)")
        except Exception as e:
            logger.warning(f"[restore] Failed to restore reference: {e}")

    # Restore chat history
    if user_id not in _chat_history or not _chat_history[user_id]:
        try:
            chats = brain.query(topic="user_session_chat", source=str(user_id), limit=1)
            if chats:
                hist = json.loads(chats[0]["content"])
                # Validate: strip trailing tool_use without matching tool_result
                # The API requires every tool_use to have a tool_result immediately after
                hist = _sanitize_chat_history(hist)
                _chat_history[user_id] = hist
                logger.info(f"[restore] Restored chat history for user {user_id} ({len(hist)} messages)")
        except Exception as e:
            logger.warning(f"[restore] Failed to restore chat history: {e}")

    # Restore last generated post (must come after reference restore for cross-linking)
    _restore_user_post(user_id)

# ── Sliding conversation memory ─────────────────────────
# Keeps recent posts and analyses available for ~15 messages, auto-prunes
from state import MEMORY_TTL_MESSAGES, _message_counter, _conversation_memory


def _bump_message_counter(user_id: int) -> int:
    """Increment message counter and prune old memory entries."""
    _message_counter[user_id] = _message_counter.get(user_id, 0) + 1
    count = _message_counter[user_id]

    # Prune entries older than MEMORY_TTL_MESSAGES
    if user_id in _conversation_memory:
        _conversation_memory[user_id] = [
            entry for entry in _conversation_memory[user_id]
            if count - entry["msg_num"] < MEMORY_TTL_MESSAGES
        ]
    return count


def _get_memory_context(user_id: int) -> str:
    """Build a summary of conversation memory for the system prompt."""
    if user_id not in _conversation_memory or not _conversation_memory[user_id]:
        return ""

    current_msg = _message_counter.get(user_id, 0)
    lines = ["\nConversation memory (recent items, auto-expires):"]
    for entry in _conversation_memory[user_id]:
        age = current_msg - entry["msg_num"]
        freshness = "just now" if age <= 1 else f"{age} msgs ago"
        if entry["type"] == "analysis":
            analysis = entry["data"]
            lines.append(f"  📸 Reference analysis ({freshness}): {entry['label']}")
        elif entry["type"] == "post":
            post = entry["data"]
            d = post.get("decisions")
            client = post.get("client", "?")
            headline = d.headline if d else "?"
            lines.append(f"  🖼️ Generated post ({freshness}): {client} — \"{headline}\"")

    return "\n".join(lines)


def _add_to_history(user_id: int, role: str, content):
    """Add a message to user's conversation history.
    Content can be a string or a list of content blocks (for tool_use)."""
    if user_id not in _chat_history:
        _chat_history[user_id] = []
    _chat_history[user_id].append({"role": role, "content": content})
    # Trim but never split tool_use/tool_result pairs
    _trim_history(user_id)
    # Persist every 3 messages (not every message — too many DB writes)
    if len(_chat_history[user_id]) % 3 == 0:
        _persist_chat_history(user_id)


def _trim_history(user_id: int):
    """Strict rolling window: always keep exactly MAX_HISTORY messages.
    Old messages are simply dropped — no tool_use IDs survive = no corruption."""
    hist = _chat_history.get(user_id, [])
    if len(hist) <= MAX_HISTORY:
        return

    # Drop from the front until we're at MAX_HISTORY
    # But never start on a tool_result (it needs the preceding tool_use)
    while len(hist) > MAX_HISTORY:
        first = hist[0]
        # If the first message is a user tool_result, drop it (orphaned anyway)
        # If the first message is an assistant with tool_use, drop it AND the next tool_result
        if first.get("role") == "user" and isinstance(first.get("content"), list):
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in first["content"]
            )
            if has_tool_result:
                # This is an orphaned tool_result — drop it
                hist.pop(0)
                continue

        hist.pop(0)

    # Safety: must start with user message
    while hist and hist[0].get("role") != "user":
        hist.pop(0)

    _chat_history[user_id] = hist
    logger.debug(f"[trim] user {user_id}: trimmed to {len(hist)} messages")


def _get_history_for_api(user_id: int) -> list[dict]:
    """Get conversation history formatted for the Anthropic API.
    Enforces the API's strict adjacency rules:
    - Must start with a user message (no tool_results in first message)
    - Each tool_result must reference a tool_use in the IMMEDIATELY PREVIOUS assistant message
    - Each assistant tool_use must have a matching tool_result in the IMMEDIATELY NEXT user message
    - Roles must alternate: user, assistant, user, assistant, ...
    """
    hist = _chat_history.get(user_id, [])
    if not hist:
        return []

    # Step 1: walk messages pairwise, validate tool_use↔tool_result adjacency
    cleaned = []
    for msg in hist:
        role = msg.get("role")
        content = msg.get("content", [])

        if role == "user" and isinstance(content, list):
            # Get tool_use IDs from the immediately previous assistant message
            prev_tool_use_ids = set()
            if cleaned and cleaned[-1].get("role") == "assistant":
                prev_content = cleaned[-1].get("content", [])
                if isinstance(prev_content, list):
                    for block in prev_content:
                        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                            prev_tool_use_ids.add(block["id"])

            # Filter: only keep tool_results that match the previous assistant's tool_use IDs
            filtered = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if block.get("tool_use_id") not in prev_tool_use_ids:
                        logger.warning(f"[api-sanitize] Dropping tool_result for {block.get('tool_use_id')} — no matching tool_use in previous message")
                        continue
                filtered.append(block)

            if not filtered:
                continue  # skip empty messages
            cleaned.append({**msg, "content": filtered})

        elif role == "assistant" and isinstance(content, list):
            # Check if this assistant message has tool_use blocks
            tool_use_ids_here = set()
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                    tool_use_ids_here.add(block["id"])

            # We'll add it now; if the next user message doesn't have matching
            # tool_results, we'll retroactively strip the tool_use blocks later
            cleaned.append(msg)
        else:
            cleaned.append(msg)

    # Step 2: retroactively strip assistant tool_use blocks without matching next tool_results
    final = []
    for i, msg in enumerate(cleaned):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            tool_use_ids = {
                b["id"] for b in msg["content"]
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
            }
            if tool_use_ids:
                # Check the next message for matching tool_results
                next_msg = cleaned[i + 1] if i + 1 < len(cleaned) else None
                next_result_ids = set()
                if next_msg and next_msg.get("role") == "user" and isinstance(next_msg.get("content"), list):
                    for b in next_msg["content"]:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            next_result_ids.add(b.get("tool_use_id"))
                orphan_ids = tool_use_ids - next_result_ids
                if orphan_ids:
                    # Strip orphaned tool_use blocks from this message
                    filtered = [
                        b for b in msg["content"]
                        if not (isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") in orphan_ids)
                    ]
                    if not filtered:
                        continue  # skip entirely empty assistant message
                    msg = {**msg, "content": filtered}
        final.append(msg)

    # Step 3: ensure starts with user, alternating roles
    while final and final[0].get("role") != "user":
        final.pop(0)

    # Merge consecutive same-role messages
    merged = []
    for msg in final:
        if merged and merged[-1].get("role") == msg.get("role"):
            prev_c = merged[-1].get("content", [])
            curr_c = msg.get("content", [])
            if isinstance(prev_c, list) and isinstance(curr_c, list):
                merged[-1] = {**merged[-1], "content": prev_c + curr_c}
            elif isinstance(prev_c, str) and isinstance(curr_c, str):
                merged[-1] = {**merged[-1], "content": prev_c + "\n" + curr_c}
        else:
            merged.append(msg)

    # Step 4: FINAL re-validation — Step 2 may have removed assistants that Step 1
    # already validated tool_results against, leaving orphaned tool_results after merge.
    validated = []
    for msg in merged:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            prev_tool_use_ids = set()
            if validated and validated[-1].get("role") == "assistant":
                prev_content = validated[-1].get("content", [])
                if isinstance(prev_content, list):
                    for block in prev_content:
                        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                            prev_tool_use_ids.add(block["id"])
            filtered = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if block.get("tool_use_id") not in prev_tool_use_ids:
                        logger.warning(f"[final-sanitize] Dropping stale tool_result {block.get('tool_use_id')}")
                        continue
                filtered.append(block)
            if not filtered:
                continue
            validated.append({**msg, "content": filtered})
        else:
            validated.append(msg)

    return validated


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

        # If post was generated from a reference, show the element descriptions
        # so Claude knows what was in the inspiration (e.g. "woman portrait", "gear icon")
        ref_elements = post_data.get("reference_elements")
        if ref_elements:
            elem_lines = []
            for elem in ref_elements:
                slot = elem.get("slot", "?")
                desc = elem.get("description", elem.get("prompt", "?"))
                sourcing = elem.get("sourcing", "ai_photo")
                is_bg = elem.get("is_background", False)
                role = "background" if is_bg else f"IMAGE_{slot}"
                elem_lines.append(f"    {role}: {desc}")
            post_context += (
                f"\n  Reference elements (from inspiration image — use these descriptions for add_element):\n"
                + "\n".join(elem_lines) + "\n"
            )

        # Also include taste analysis summary if available
        analysis = _last_analysis_by_user.get(user_id)
        if analysis and not ref_elements:
            comp = analysis.get("composition", {})
            feeling = analysis.get("feeling", {})
            what_works = analysis.get("what_makes_it_work", "")
            if comp or what_works:
                post_context += (
                    f"\n  Inspiration analysis (what the user sent):\n"
                    f"    Composition: {comp.get('template_match', '?')}, {comp.get('text_position', '?')} text\n"
                    f"    Mood: {feeling.get('mood', '?')}\n"
                    f"    What works: {what_works[:150]}\n"
                )

    has_pending_scout = user_id in _pending_scout
    scout_context = ""
    if has_pending_scout:
        pending = _pending_scout[user_id]
        items = pending.get("items", [])
        item_nums = ", ".join(str(i["index"]) for i in items)
        scout_context = (
            f"\n⚠️ PENDING SCOUT: User has {len(items)} design references waiting for approval (items: {item_nums}). "
            f"If the user replies with numbers, 'all', or 'none', call approve_scout. "
            f"Numbers like '1, 3, 5' → approve_scout with selected=[1,3,5]. "
            f"'all' → approve_scout with all=true. 'none' or 'skip' → approve_scout with selected=[]."
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
- HONESTY: NEVER say you did something unless a tool was actually called. If you can't do what the user asks, say so honestly. Don't say "Done!" or "I've added that" unless a tool executed and returned a new image.
- ADDING ELEMENTS: When user wants to ADD a visual element to an existing post (a person, object, photo, icon), use edit_post with BOTH feedback (WHERE to place it) AND add_element (WHAT to add, e.g. "the woman from the inspiration", "a product shot"). Opus will look at the reference image and write the generation prompt. Without add_element, the image won't be generated.
- COLOR CHANGES: When user says "replace X with Y", change ALL fields containing that color (color_bg, color_accent, color_text, color_subtext). Use DISTINCT, clearly different hex values — never subtle variations.
  Reference: Red=#DC2626, Orange=#EA580C, Amber=#D97706, Yellow=#EAB308, Lime=#65A30D, Green=#16A34A, Emerald=#059669, Teal=#0D9488, Cyan=#0891B2, Sky=#0284C7, Blue=#2563EB, Indigo=#4F46E5, Violet=#7C3AED, Purple=#9333EA, Fuchsia=#C026D3, Pink=#DB2777, Rose=#E11D48, White=#FFFFFF, Black=#000000, Gray=#6B7280, Beige=#F5F0E8, Navy=#1E3A5F, Burgundy=#800020, Gold=#FFD700, Coral=#FF6B6B, Turquoise=#40E0D0, Peach=#FFCBA4, Lavender=#E6E6FA, Mint=#98FB98, Cream=#FFFDD0, Charcoal=#36454F
- STOCK PHOTOS: When user asks to "use stock photos", "use real photos", "no AI images", "use actual photos", or anything similar → set image_source="stock" on generate_post. This is CRITICAL. Default is "auto".
- LANDSCAPE FORMAT: When user asks for "landscape", "16:9", "widescreen", "horizontal" or "can you make it landscape" → set format="landscape" on generate_post. Default is "square".
- ONE TOOL PER EDIT: When the user gives feedback on the current post, use ONLY ONE tool call. Combine everything into a single edit_post. Do NOT split a single request into edit_post + replace_image. Example: "make the woman bigger and darker orange" = ONE edit_post call with feedback + color_bg. Only use replace_image ALONE when the user SPECIFICALLY wants a brand new image or photo.
- BACKGROUND CHANGES: Use replace_image with background_style ONLY when the user explicitly asks for a gradient, texture, pattern, or abstract AI-generated background (e.g. "make a gradient background", "abstract dark texture", "smoky background effect"). For simple color changes like "darker orange", "lighter", "change to blue", "warmer color" → use edit_post with color_bg. color_bg works for ANY solid/flat color.
- INSPIRATION / COPY: When user sends an image and says "make a post like this", "similar to this one", "based on this", "copy this", "recreate this", "clone this design" → set use_last_inspiration=true on generate_post. The system will decompose the image element-by-element (icons, UI widgets, photos, shapes) and replicate it precisely. Do NOT set this for normal posts.
- CLIENT RULES: When user says "never use X for client", "always use Y for client", "client should not have Z" → call save_client_rule to permanently store this. Do this IN ADDITION to any other action (like generating a new post). Example: "never use orange for Georgoulis" → save_client_rule + generate_post.
- DESIGN SCOUT: When user asks to find fresh designs, inspiration, or specific styles, call scout_designs. Always set the focus field from what they say — e.g. "find me dark luxury posts" → focus="dark luxury editorial", "look for minimal health brand posts" → focus="minimal health brand", "something with bold typography" → focus="bold typography". If there are pending scout results, watch for the user's approval reply.
- GENERATE FROM SCOUT: When user says "make a post like number 3" or "use layout 2 for LMW" while scout results are pending, call generate_from_scout with the item number, client and brief. This generates a post using that specific layout as a blueprint.
{scout_context}
{_get_memory_context(user_id)}"""


# ── Photo handler (taste ingestion + logo detection) ─────
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming photos — Claude decides if it's a logo/brand asset or inspiration."""
    msg = update.message
    if not msg or not msg.photo:
        return

    photo = msg.photo[-1]
    caption = msg.caption or ""
    user_id = msg.from_user.id if msg.from_user else 0
    _restore_user_session(user_id)
    _bump_message_counter(user_id)
    history_text = _get_history_text_simple(user_id)

    status_msg = await msg.reply_text("🔍 Looking at this...")

    try:
        file = await photo.get_file()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        await file.download_to_drive(str(tmp_path))

        # Step 1: Claude classifies what this image is
        has_recent_post = user_id in _last_post_by_user
        image_intent = await _classify_image(tmp_path, caption, history_text, has_recent_post=has_recent_post)
        logger.info(f"[photo] classified as: {image_intent}")

        if image_intent.get("type") == "logo":
            # ── LOGO / BRAND ASSET ──
            await _handle_logo_upload(msg, status_msg, tmp_path, caption, image_intent, user_id)
            return

        if image_intent.get("type") == "edit_feedback" and has_recent_post:
            # ── EDIT FEEDBACK WITH SCREENSHOT ──
            await _handle_edit_feedback_with_screenshot(msg, status_msg, tmp_path, caption, image_intent, user_id)
            return

        # ── INSPIRATION (default) ──
        client = image_intent.get("client") or _extract_client_from_caption(caption)

        # Detect if user wants to GENERATE (not just share inspiration)
        _generation_keywords = re.compile(
            r'\b(make|create|copy|recreate|clone|replicate|build|design|generate|κάνε|φτιάξε|δημιούργησε)\b',
            re.IGNORECASE,
        )
        wants_generation = bool(caption and _generation_keywords.search(caption))

        # Save image bytes immediately — generation needs them
        _last_analysis_by_user[user_id] = {}
        try:
            img_b64 = base64.b64encode(tmp_path.read_bytes()).decode("utf-8")
            _last_analysis_by_user[user_id]["_image_b64"] = img_b64
        except Exception as e:
            logger.debug(f"[photo] Could not attach image b64 to analysis: {e}")

        # Save image and let Claude decide what to do — no analysis, no breakdown
        _persist_user_reference(user_id)
        await status_msg.delete()

        _add_to_history(user_id, "user", f"[sent a reference image] {caption}")

        if caption:
            # User sent a photo with a caption — let Claude decide (generate, edit, etc)
            logger.info(f"[photo] Image with caption — letting Claude decide: '{caption[:60]}'")
            request_text = (
                f"[User sent a reference image. The image is saved as the latest inspiration. "
                f"The user's message: \"{caption}\"]\n\n"
                f"If the user wants to generate/create/make/remake/copy a post based on this image, "
                f"call generate_post with use_last_inspiration=true. "
                f"Extract the client name and brief from the caption. "
                f"If they're just sharing or adding context, respond naturally and briefly."
            )
        else:
            # No caption — just acknowledge the image
            logger.info(f"[photo] Image saved, no caption")
            request_text = (
                f"[User sent a reference image with no caption. The image is saved. "
                f"Acknowledge briefly — say something like 'Got it, saved as reference. "
                f"Want me to make a post based on this?']"
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


async def _classify_image(image_path: Path, caption: str, history: str, has_recent_post: bool = False) -> dict:
    """Use Claude Vision to classify if an image is a logo, inspiration, or edit feedback."""
    img_bytes = image_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    suffix = image_path.suffix.lower()
    media = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    recent_post_hint = ""
    if has_recent_post:
        recent_post_hint = (
            "\nIMPORTANT: The user has a recently generated post. If this image looks like a screenshot "
            "of a design/post (possibly cropped or annotated) and the caption sounds like feedback "
            "(e.g. 'remove this', 'change this part', 'make this bigger', 'delete the text here', "
            "'this element is wrong'), classify it as \"edit_feedback\".\n"
        )

    try:
        response = _ai_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=(
                "You classify images sent to a design bot. What is this image?\n\n"
                "Three types:\n"
                "1. \"logo\" — a logo, icon, or brand mark\n"
                "2. \"inspiration\" — a design reference the user likes (for learning taste)\n"
                "3. \"edit_feedback\" — a screenshot of the bot's own output (or a cropped part of it) "
                "where the user is pointing out something to change, remove, or fix\n\n"
                "Consider the caption and conversation history for context.\n"
                f"{recent_post_hint}"
                "Return JSON: {\"type\": \"logo\" or \"inspiration\" or \"edit_feedback\", "
                "\"client\": \"ClientName\" or null, \"reason\": \"brief reason\", "
                "\"edit_instruction\": \"what the user wants changed\" or null}\n"
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


async def _handle_edit_feedback_with_screenshot(
    msg, status_msg, image_path: Path, caption: str, intent: dict, user_id: int,
):
    """User sent a screenshot of the bot's output with edit instructions.

    The screenshot is passed to the edit flow so Opus can see exactly
    what the user is pointing at.
    """
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    # Store the screenshot so the edit flow can use it
    if user_id not in _last_post_by_user:
        await status_msg.delete()
        await msg.reply_text("I don't have a recent post to edit. Generate one first!")
        return

    # Attach the user's screenshot to the post data so _exec_edit_post can use it
    _last_post_by_user[user_id]["user_screenshot_b64"] = img_b64

    feedback = caption or intent.get("edit_instruction") or "Fix what I'm pointing at in this screenshot"
    logger.info(f"[photo] Edit feedback with screenshot: {feedback[:80]}")

    await status_msg.delete()

    # Route through the tool-use loop as an edit request
    _add_to_history(user_id, "user", f"[sent screenshot of my design with edit instructions] {feedback}")

    request_text = (
        f"[The user sent a SCREENSHOT of the current post and wants changes. "
        f"The screenshot has been saved — the edit tool will pass it to Opus. "
        f"The user's instruction: \"{feedback}\"]\n\n"
        f"Call edit_post with feedback set to the user's instruction. "
        f"Do NOT treat this as inspiration or a new generation request."
    )
    _add_to_history(user_id, "user", request_text)
    await _run_tool_use_loop(user_id, msg)

    # Clean up temp file
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
    elif tool_name == "restore_image":
        return await _exec_restore_image(tool_input, user_id, msg)
    elif tool_name == "save_favorite":
        return await _exec_save_favorite(tool_input, user_id, msg)
    elif tool_name == "delete_template":
        return await _exec_delete_template(tool_input, user_id, msg)
    elif tool_name == "process_feedback":
        return await _exec_process_feedback(tool_input, user_id)
    elif tool_name == "get_taste_profile":
        return await _exec_get_taste(user_id, msg)
    elif tool_name == "manage_templates":
        return await _exec_manage_templates(tool_input, msg)
    elif tool_name == "save_client_rule":
        return await _exec_save_client_rule(tool_input, user_id, msg)
    elif tool_name == "resend_last_post":
        return await _exec_resend(user_id, msg)
    elif tool_name == "scout_designs":
        return await _exec_scout_designs(tool_input, user_id, msg)
    elif tool_name == "approve_scout":
        return await _exec_approve_scout(tool_input, user_id, msg)
    elif tool_name == "generate_from_scout":
        return await _exec_generate_from_scout(tool_input, user_id, msg)
    else:
        return f"Unknown tool: {tool_name}"


# ── Tool handlers live in their own modules (imported after helpers are defined) ──
from exec_generate import (
    _exec_generate_post,
    _exec_generate_carousel,
    _exec_replace_image,
    _exec_restore_image,
)
from exec_edit import _exec_edit_post
from exec_scout import (
    _exec_scout_designs,
    _exec_approve_scout,
    _exec_generate_from_scout,
)
from exec_manage import (
    _exec_save_favorite,
    _exec_delete_template,
    _exec_process_feedback,
    _exec_get_taste,
    _exec_manage_templates,
    _exec_save_client_rule,
    _exec_resend,
)


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

            # Guard: detect conflicting tool calls in the same turn
            tool_names_this_turn = [b.name for b in response.content if b.type == "tool_use"]
            has_edit = "edit_post" in tool_names_this_turn
            has_replace = "replace_image" in tool_names_this_turn
            skip_replace = has_edit and has_replace  # edit_post already handles it

            for block in response.content:
                if block.type == "tool_use":
                    if skip_replace and block.name == "replace_image":
                        logger.warning(f"[tool-use] Skipping replace_image — edit_post already called in same turn")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Skipped — edit_post already handles this request. No second post needed.",
                        })
                        continue
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

    except anthropic.BadRequestError as e:
        error_msg = str(e)
        logger.error(f"Tool-use handler failed (400): {error_msg}")
        if "tool_result" in error_msg or "tool_use" in error_msg:
            # Corrupted chat history — nuke it so next message works
            logger.warning(f"[recovery] Clearing corrupted chat history for user {user_id}")
            _chat_history[user_id] = []
            try:
                brain.store(
                    content="[]",
                    topic="user_session_chat",
                    source=str(user_id),
                    summary="chat history (cleared after corruption)",
                )
            except Exception:
                pass
            try:
                await msg.reply_text("🔄 I had a memory glitch — cleared my conversation history. Please try again!")
            except Exception:
                pass
        else:
            try:
                await msg.reply_text("🎨 Hey! Something went wrong. Try again or say 'make a post for [client]'.")
            except Exception:
                pass
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

    # Restore session from DB if this is first message after restart
    _restore_user_session(user_id)

    # Check if the user is replying to an old post (to save/edit/reuse it)
    replied_post = None
    if msg.reply_to_message:
        reply_msg_id = msg.reply_to_message.message_id
        replied_post = _posts_by_msg_id.get(reply_msg_id)
        if replied_post:
            # Load this old post as the "current" post so save/edit tools work on it
            post_data = {k: v for k, v in replied_post.items() if k != "_user_id"}
            _last_post_by_user[user_id] = post_data
            logger.info(f"[text] User replied to tracked post msg_id={reply_msg_id} — loaded as current post")

            # Also save the old post's rendered image as the latest reference
            # so "make this" / "use this template" triggers copy-mode generation
            rendered_path = post_data.get("rendered_path")
            if rendered_path and Path(rendered_path).exists():
                try:
                    img_b64 = base64.b64encode(Path(rendered_path).read_bytes()).decode("utf-8")
                    _last_analysis_by_user[user_id] = {"_image_b64": img_b64}
                    logger.info(f"[text] Loaded rendered image as reference for reuse")
                except Exception as e:
                    logger.debug(f"[text] Could not load rendered image as reference: {e}")

            client = replied_post.get('client', '?')
            text = (
                f"[Replying to a previous post for {client}. "
                f"The post's rendered image is now stored as the latest reference. "
                f"If the user wants to generate a new post using this style/template, "
                f"call generate_post with use_last_inspiration=true and client='{client}'.] {text}"
            )

    _bump_message_counter(user_id)
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

    logger.info("Bot ready. Polling for Telegram updates...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
