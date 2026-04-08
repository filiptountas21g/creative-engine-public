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
MAX_HISTORY = 10  # tight window — prevents corruption from complex tool_use/tool_result pairs
_chat_history: dict[int, list[dict]] = {}

# Track last pipeline result per user for edits (in-memory only, resets on restart)
_last_post_by_user: dict[int, dict] = {}

# Track posts by Telegram message ID — so user can reply to any old post to save/edit it
# Key: message_id → post_data dict (same format as _last_post_by_user values)
MAX_TRACKED_POSTS = 20  # keep last 20 posts per user in memory
_posts_by_msg_id: dict[int, dict] = {}

# Image vault — stores all images ever used in posts so they can be restored
# Key: user_id → {"hero": ImageResult, "extra_1": ImageResult, ...}
# Images accumulate across edits — never lost until explicit clear
MAX_VAULT_IMAGES = 10
_image_vault: dict[int, dict[str, object]] = {}

# Track previous decisions per user for variety
_previous_decisions: dict[int, list[dict]] = {}

# Pending scout results awaiting user approval
_pending_scout: dict[int, dict] = {}

# Track which users have been restored from DB this session
_restored_users: set[int] = set()


MAX_SAVED_REFERENCES = 5  # keep last 5 references per user (~1MB total)


def _vault_save_images(user_id: int, hero_image=None, extra_images=None):
    """Save all images from a post to the vault. Accumulates — never overwrites."""
    if user_id not in _image_vault:
        _image_vault[user_id] = {}

    vault = _image_vault[user_id]

    if hero_image and hasattr(hero_image, 'image_path') and hero_image.image_path:
        # Save with a descriptive key based on the prompt
        label = (hero_image.prompt_used or "hero")[:60].strip()
        vault[f"hero_{label}"] = hero_image
        # Always keep a "latest_hero" reference
        vault["latest_hero"] = hero_image
        logger.debug(f"[vault] Saved hero: {label[:30]}")

    if extra_images:
        for i, img in enumerate(extra_images):
            if img and hasattr(img, 'image_path') and img.image_path:
                label = (img.prompt_used or f"extra_{i+1}")[:60].strip()
                vault[f"extra_{i+1}_{label}"] = img
                vault[f"latest_extra_{i+1}"] = img
                logger.debug(f"[vault] Saved extra_{i+1}: {label[:30]}")

    # Prune if too many
    if len(vault) > MAX_VAULT_IMAGES * 2:
        # Keep only "latest_" keys and the most recent others
        latest = {k: v for k, v in vault.items() if k.startswith("latest_")}
        others = [(k, v) for k, v in vault.items() if not k.startswith("latest_")]
        others = others[-(MAX_VAULT_IMAGES):]
        _image_vault[user_id] = {**dict(others), **latest}

    logger.info(f"[vault] User {user_id}: {len(_image_vault[user_id])} images stored")


def _vault_get(user_id: int, search: str = None):
    """Get an image from the vault. If search is given, fuzzy match against labels."""
    vault = _image_vault.get(user_id, {})
    if not vault:
        return None

    if not search:
        return vault.get("latest_hero")

    search_lower = search.lower()
    # Try exact key match first
    for key, img in vault.items():
        if search_lower in key.lower():
            return img

    # Try matching against prompt_used
    for key, img in vault.items():
        if hasattr(img, 'prompt_used') and img.prompt_used and search_lower in img.prompt_used.lower():
            return img

    # Fall back to latest hero
    return vault.get("latest_hero")


def _track_post_by_msg_id(msg_id: int, user_id: int, post_data: dict):
    """Track a sent post by its Telegram message ID so user can reply to it later."""
    _posts_by_msg_id[msg_id] = {**post_data, "_user_id": user_id}
    # Prune old entries — keep only last MAX_TRACKED_POSTS per user
    user_posts = [(mid, d) for mid, d in _posts_by_msg_id.items() if d.get("_user_id") == user_id]
    if len(user_posts) > MAX_TRACKED_POSTS:
        # Sort by message ID (older = lower) and remove excess
        user_posts.sort(key=lambda x: x[0])
        for mid, _ in user_posts[:-MAX_TRACKED_POSTS]:
            _posts_by_msg_id.pop(mid, None)
    logger.debug(f"[track] Post tracked: msg_id={msg_id}, total tracked={len(_posts_by_msg_id)}")


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
                except Exception:
                    pass
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


def _persist_user_post(user_id: int):
    """Save the user's last generated post to Brain (survives restarts).
    Stores metadata + template in one row, image files in separate rows."""
    if user_id not in _last_post_by_user:
        return
    post_data = _last_post_by_user[user_id]

    try:
        from dataclasses import asdict
        import base64 as _b64

        # Build serializable metadata (text only, no large blobs)
        meta = {
            "client": post_data.get("client", ""),
            "canvas_format": post_data.get("canvas_format", "square"),
            "template_html": post_data.get("template_html", ""),
            "reference_elements": post_data.get("reference_elements"),
            "has_reference_image": bool(post_data.get("reference_image_b64")),
        }

        if post_data.get("decisions"):
            meta["decisions"] = asdict(post_data["decisions"])
        if post_data.get("concept"):
            meta["concept"] = asdict(post_data["concept"])

        # Store metadata (delete old first — 1 row per user)
        brain.delete_by_topic_source("user_session_post", str(user_id))
        brain.store(
            topic="user_session_post",
            source=str(user_id),
            content=json.dumps(meta, default=str),
            summary=f"Post for {meta['client']}: {meta.get('decisions', {}).get('headline', '')[:40]}",
            tags=["session", "post", str(user_id)],
        )

        # Store image files as separate entries
        brain.delete_by_topic_source("user_session_post_images", str(user_id))

        all_images = []
        if post_data.get("image"):
            all_images.append(("hero", post_data["image"]))
        for i, ei in enumerate(post_data.get("extra_images") or []):
            all_images.append((f"extra_{i+1}", ei))

        for label, img_result in all_images[:4]:  # max 4 images to keep DB reasonable
            try:
                img_path = Path(img_result.image_path)
                if img_path.exists():
                    img_b64 = _b64.b64encode(img_path.read_bytes()).decode("utf-8")
                    brain.store(
                        topic="user_session_post_images",
                        source=str(user_id),
                        content=json.dumps({
                            "label": label,
                            "image_path": img_result.image_path,
                            "image_url": getattr(img_result, "image_url", ""),
                            "model_used": getattr(img_result, "model_used", ""),
                            "prompt_used": getattr(img_result, "prompt_used", ""),
                            "bytes_b64": img_b64,
                        }),
                        summary=f"{label}: {img_result.image_path}",
                        tags=["session", "post_image", str(user_id), label],
                    )
                    logger.info(f"[persist] Saved {label} image ({len(img_b64)//1024}KB)")
            except Exception as e:
                logger.warning(f"[persist] Failed to save {label} image: {e}")

        logger.info(f"[persist] Saved post for user {user_id} ({len(all_images)} images)")
    except Exception as e:
        logger.warning(f"[persist] Failed to save post: {e}")


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
            from pipeline.types import CreativeConcept
            concept = CreativeConcept(**meta["concept"])

        # Restore image files from DB → disk
        import base64 as _b64
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
                    p.write_bytes(_b64.b64decode(bytes_b64))

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
            from pipeline.orchestrator import _get_client_logo
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
MEMORY_TTL_MESSAGES = 15

_message_counter: dict[int, int] = {}
_conversation_memory: dict[int, list[dict]] = {}  # list of {"type", "data", "msg_num", "label"}


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


def _remember(user_id: int, entry_type: str, data: dict, label: str = ""):
    """Store something in conversation memory with the current message number."""
    if user_id not in _conversation_memory:
        _conversation_memory[user_id] = []

    _conversation_memory[user_id].append({
        "type": entry_type,  # "analysis", "post", "reference"
        "data": data,
        "msg_num": _message_counter.get(user_id, 0),
        "label": label,
    })

    # Cap at 10 entries max per user
    if len(_conversation_memory[user_id]) > 10:
        _conversation_memory[user_id] = _conversation_memory[user_id][-10:]


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

    return merged


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
                "format": {
                    "type": "string",
                    "enum": ["square", "landscape"],
                    "description": "Canvas format. 'square' = 1080x1080 (default, Instagram/LinkedIn). 'landscape' = 1920x1080 (16:9, presentations, LinkedIn documents). Use 'landscape' when the user asks for landscape, 16:9, widescreen, or horizontal format.",
                },
                "use_last_inspiration": {
                    "type": "boolean",
                    "description": "Set to true ONLY when the user explicitly asks to make a post 'like this', 'similar to this', 'based on this' referring to an inspiration image they just sent. This forces the template to replicate that specific layout. Do NOT set this when generating a normal post.",
                },
                "style_overrides": {
                    "type": "object",
                    "description": (
                        "Override specific design decisions AFTER the AI picks them. Use when the user "
                        "specifies constraints like 'use black and white', 'use Montserrat font', 'make it red'. "
                        "Any field from edit_post works here: color_bg, color_text, color_accent, color_subtext, "
                        "font_headline, font_headline_weight, font_headline_size, font_headline_case, etc."
                    ),
                    "properties": {
                        "color_bg": {"type": "string"},
                        "color_text": {"type": "string"},
                        "color_accent": {"type": "string"},
                        "color_subtext": {"type": "string"},
                        "font_headline": {"type": "string"},
                        "font_headline_weight": {"type": "integer"},
                        "font_headline_size": {"type": "integer"},
                        "font_headline_case": {"type": "string"},
                        "font_subtext": {"type": "string"},
                    },
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
            "add/remove logo, adjust background color. The hero image stays the same. "
            "Use for ANY tweak to the existing post: translating, restyling, changing text, adjusting layout, "
            "changing colors (including background color). "
            "For visual/layout feedback like 'make pictures bigger', 'move text to the right', 'more spacing', "
            "'make the image cover the background', 'darker color' — use the 'feedback' field. "
            "IMPORTANT: Only include fields that need to change. Combine ALL user requests into ONE call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feedback": {
                    "type": "string",
                    "description": (
                        "Freeform visual feedback from the user that requires HTML/CSS changes. "
                        "Use for layout tweaks like 'make the pictures bigger', 'move text to the right', "
                        "'add more space between elements', 'make the grid 2x2', etc. "
                        "Opus will directly modify the template HTML/CSS based on this feedback."
                    ),
                },
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
                "color_bg": {"type": "string", "description": "Background color (hex). Use for any color change — 'darker orange', 'change to blue', 'lighter', etc."},
                "color_text": {"type": "string", "description": "Text color (hex)"},
                "color_accent": {"type": "string", "description": "Accent color (hex)"},
                "color_subtext": {"type": "string", "description": "Subtext color (hex)"},
                "headline_margin_x": {"type": "integer", "description": "Horizontal margin (0-200px)"},
                "headline_margin_y": {"type": "integer", "description": "Vertical margin (0-200px)"},
                "headline_max_width": {"type": "string", "description": "Max width CSS (e.g. '75%')"},
                "image_padding": {"type": "integer", "description": "Image padding (0-200px)"},
                "add_logo": {"type": "boolean", "description": "Set true to add the client's logo"},
                "remove_logo": {"type": "boolean", "description": "Set true to remove the logo"},
                "remove_background": {"type": "boolean", "description": "Set true to remove the background from the hero image (make it transparent). Use when user says 'remove the background', 'cut out the person', 'transparent background', etc."},
                "add_element": {
                    "type": "string",
                    "description": (
                        "Description of a visual element to ADD to the design. "
                        "Use when the user wants to add a photo/person/object that's NOT already in the post. "
                        "Just describe WHAT — e.g. 'the woman from the inspiration', 'a portrait', 'a product photo'. "
                        "Opus will look at the reference image and write the AI generation prompt itself. "
                        "Pair with feedback for WHERE to place it — e.g. 'on the right side, 40% width'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "replace_image",
        "description": (
            "Generate a completely NEW hero image for the last post. Keeps all design decisions (font, colors, layout, text). "
            "ONLY use when the user explicitly wants a different photo/image — 'replace the photo', 'use a stock image', 'change the picture'. "
            "Do NOT use for color changes, layout tweaks, or resizing — those go to edit_post. "
            "Use background_style ONLY when user explicitly asks for AI gradient/texture/abstract background — "
            "e.g. 'make an abstract gradient background', 'smoky texture background'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "REQUIRED when the user describes what they want. What the new image should show — "
                        "e.g. 'skyscrapers at sunset', 'business people in a meeting', 'abstract tech pattern'. "
                        "Extract this directly from the user's message. If not provided, reuses the original concept."
                    ),
                },
                "image_source": {
                    "type": "string",
                    "enum": ["auto", "stock", "ai"],
                    "description": "Image source preference. 'stock' = stock photos only, 'ai' = AI-generated only, 'auto' = try stock first then AI. Use 'stock' when user asks for stock/real photos.",
                },
                "background_style": {
                    "type": "string",
                    "description": (
                        "Custom description for AI background generation. Use when the user wants a specific "
                        "background style like 'warm orange gradient', 'deep blue abstract', 'dark smoky texture', "
                        "'amber glow', etc. This overrides the concept and forces AI generation with this exact style. "
                        "Example: 'rich orange gradient with warm glow, abstract, no text'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "restore_image",
        "description": (
            "Restore a previously used image from the vault. Use when the user says "
            "'put back the woman', 'restore the old picture', 'use the previous image', "
            "'bring back the photo you removed'. The vault keeps all images from this session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Description of which image to restore — e.g. 'woman with bob cut', 'the first hero image', 'the portrait'. Fuzzy matched against stored image prompts.",
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
        "name": "delete_template",
        "description": (
            "Delete a saved/liked template from memory. Use when the user doesn't like the current style, "
            "says 'delete this template', 'remove this style', 'I don't like this', 'forget this design'. "
            "Deletes the liked template that matches the current post's style."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Only delete templates for this client. Optional."},
                "delete_all": {"type": "boolean", "description": "Delete ALL liked templates. Only if user explicitly asks."},
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
        "name": "save_client_rule",
        "description": (
            "Save a permanent design rule or preference for a specific client. "
            "Use when the user says things like 'never use orange for Georgoulis', "
            "'always use dark backgrounds for LMW', 'Georgoulis should always use serif fonts'. "
            "These rules persist in the Brain and are applied to ALL future posts for that client."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client name the rule applies to"},
                "rule": {"type": "string", "description": "The design rule in clear language, e.g. 'never use orange or warm accent colors'"},
                "avoid_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific hex colors to avoid, e.g. ['#D97706', '#EA580C', '#F59E0B']",
                },
                "prefer_colors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred hex colors for this client",
                },
                "prefer_fonts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred fonts for this client",
                },
                "avoid_fonts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fonts to avoid for this client",
                },
            },
            "required": ["client", "rule"],
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
    {
        "name": "scout_designs",
        "description": (
            "Search the internet for fresh design inspiration using Perplexity. "
            "Returns a PREVIEW list of design references — does NOT save anything yet. "
            "The user must approve which ones to save (using approve_scout). "
            "Use when user says 'find me fresh designs', 'search for new layouts', "
            "'I'm bored of the layouts', 'bring me something new', 'find inspiration'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client": {
                    "type": "string",
                    "description": "Client name — searches for industry-specific designs. Use 'ALL' for general.",
                },
                "focus": {
                    "type": "string",
                    "description": "Specific focus for the search — e.g. 'minimalist healthcare', 'bold typography', 'editorial layouts with serif fonts'. Captures what the user specifically wants to find.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "approve_scout",
        "description": (
            "Approve specific design references from the scout results and save them to Brain. "
            "Use AFTER scout_designs has shown the preview list. "
            "User says which ones to keep: '1, 3, 5' or 'all' or 'none'. "
            "Only approved designs get processed and stored. "
            "If the user says WHY they don't like the results (e.g. 'none, too colorful' or "
            "'skip, I hate these layouts'), capture that reason in the feedback field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selected": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of item numbers to approve (1-based). E.g. [1, 3, 5]",
                },
                "all": {
                    "type": "boolean",
                    "description": "Set to true if user wants all items approved",
                },
                "feedback": {
                    "type": "string",
                    "description": "Why the user didn't like the results, if they said so. E.g. 'too colorful', 'not editorial enough', 'too tutorial-like'. Stored to improve future scouts.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_from_scout",
        "description": (
            "Generate a post using a specific scout design reference as the layout blueprint. "
            "Use when user says 'make a post like number 3', 'use that layout for LMW', "
            "'make one like 2 for Somamed'. Requires pending scout results from scout_designs. "
            "This both saves that specific layout AND generates a post following it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scout_item": {
                    "type": "integer",
                    "description": "The scout item number to use as layout (1-based)",
                },
                "client": {
                    "type": "string",
                    "description": "Client name for the post",
                },
                "brief": {
                    "type": "string",
                    "description": "Creative direction / topic for the post",
                },
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "instagram", "facebook"],
                },
                "image_source": {
                    "type": "string",
                    "enum": ["auto", "stock", "ai"],
                },
            },
            "required": ["scout_item", "client", "brief"],
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
                f"\n  Reference elements (from inspiration image — use these descriptions for add_image_prompt):\n"
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
        import re as _re
        _generation_keywords = _re.compile(
            r'\b(make|create|copy|recreate|clone|replicate|build|design|generate|κάνε|φτιάξε|δημιούργησε)\b',
            _re.IGNORECASE,
        )
        wants_generation = bool(caption and _generation_keywords.search(caption))

        # Save image bytes immediately — generation needs them
        import base64
        _last_analysis_by_user[user_id] = {}
        try:
            img_b64 = base64.b64encode(tmp_path.read_bytes()).decode("utf-8")
            _last_analysis_by_user[user_id]["_image_b64"] = img_b64
        except Exception:
            pass

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
    import base64
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


async def _handle_edit_feedback_with_screenshot(
    msg, status_msg, image_path: Path, caption: str, intent: dict, user_id: int,
):
    """User sent a screenshot of the bot's output with edit instructions.

    The screenshot is passed to the edit flow so Opus can see exactly
    what the user is pointing at.
    """
    import base64

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
        from dataclasses import replace as dc_replace
        overrides = {k: v for k, v in style_overrides.items() if hasattr(result.decisions, k)}
        if overrides:
            logger.info(f"Applying style overrides: {overrides}")
            result.decisions = dc_replace(result.decisions, **overrides)
            # Re-render with overridden decisions
            try:
                await status_msg.edit_text(f"🎨 Applying your style preferences...")
                from pipeline.steps.render import render as render_fn
                render_result = await render_fn(
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
            # Compress if file is too large for Telegram (max 10MB, aim for < 5MB)
            send_path = img_path
            if img_path.stat().st_size > 5 * 1024 * 1024:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as pil_img:
                    compressed = img_path.with_suffix(".jpg")
                    pil_img.convert("RGB").save(compressed, "JPEG", quality=85)
                    send_path = compressed
                    logger.info(f"Compressed image: {img_path.stat().st_size // 1024}KB → {compressed.stat().st_size // 1024}KB")
            sent_msg = await msg.reply_photo(
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
                "extra_images": getattr(last, 'extra_images', None),
                "canvas_format": "square",  # carousels are always square
            }
            _persist_user_post(user_id)


    return f"Carousel of {len(successful)} posts generated for {client_name}. Images sent."


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

        # Remove background from hero image
        if changes.pop("remove_background", None):
            if image and image.image_path:
                from pipeline.steps.image_gen import remove_background
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
        # Legacy support
        if not add_element:
            add_element = changes.pop("add_image_prompt", None)

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
                from dataclasses import replace as dc_replace
                from pipeline.steps.brain_read import brain_read
                from pipeline.steps.dynamic_template import describe_element_for_generation

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
                    element_concept = dc_replace(
                        concept,
                        object=generated_prompt,
                        why=f"Visual element: {add_element}",
                        composition_note="Single subject, clean background, suitable for compositing into a design layout",
                        what_to_avoid="text, logos, busy backgrounds, multiple subjects",
                    )
                else:
                    from pipeline.types import CreativeConcept
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
            from pipeline.steps.dynamic_template import fix_template_from_critique
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
        import re as _re
        _existing_slots = set(_re.findall(r'\{\{IMAGE_(\d+)\}\}', template_html or ""))
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
            from pipeline.steps.critique import check_edit_applied
            from pipeline.steps.dynamic_template import fix_template_from_critique

            max_verify_rounds = 2
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

                retry_critique = (
                    f"The user asked: \"{user_feedback}\"\n"
                    f"This was NOT applied. Vision says: {check.get('what_i_see', '')}\n\n"
                    f"[CRITICAL] user_feedback: {fix_instruction}\n"
                    f"  → Fix: {fix_instruction}\n"
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

        img_path = Path(render_result.final_image_path)
        sent_edit_msg = await msg.reply_photo(photo=open(img_path, "rb"), caption=result_text[:1024], read_timeout=60, write_timeout=60)

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
            from dataclasses import replace as dc_replace
            bg_concept = dc_replace(
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
            from dataclasses import replace as dc_replace
            new_concept = dc_replace(
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

        img_path = Path(render_result.final_image_path)
        prompt_hint = (restored_image.prompt_used or "previous image")[:50]
        result_text = f"✅ Restored image: {prompt_hint}"

        await status_msg.delete()
        await msg.reply_photo(photo=open(img_path, "rb"), caption=result_text[:1024], read_timeout=60, write_timeout=60)

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

    try:
        await save_liked_template(
            brain=brain,
            decisions=decisions,
            template_html=template_html,
            concept_summary=concept_summary,
            client=client_name,
            modifications=modifications,
        )

        # Also save the actual HTML to Google Drive for reuse
        from pipeline.steps.dynamic_template import save_template_to_drive
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
        await msg.reply_photo(
            photo=open(Path(rendered), "rb"),
            caption=f"📎 Last post for {post_data.get('client', '?')}",
            read_timeout=60, write_timeout=60,
        )
        return "Last post re-sent."
    else:
        await msg.reply_text("⚠️ The file is no longer available. Try generating a new one.")
        return "File not available."


async def _exec_scout_designs(params: dict, user_id: int, msg) -> str:
    """Phase 1: Search for fresh design inspiration — show preview, wait for approval."""
    from pipeline.steps.design_scout import scout_search, detect_staleness

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

        import asyncio as _asyncio
        import httpx as _httpx
        import io

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
                    async with _httpx.AsyncClient(timeout=25, follow_redirects=True) as hc:
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
            await _asyncio.sleep(0.3)  # Avoid Telegram flood limits

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
    from pipeline.steps.design_scout import scout_approve

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
    from pipeline.steps.design_scout import extract_single_reference, scout_approve

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
                img_path = Path(result.image_path)
                send_path = img_path
                if img_path.stat().st_size > 5 * 1024 * 1024:
                    from PIL import Image as PILImage
                    with PILImage.open(img_path) as pil_img:
                        compressed = img_path.with_suffix(".jpg")
                        pil_img.convert("RGB").save(compressed, "JPEG", quality=85)
                        send_path = compressed

                caption = f"🎨 Layout: {item_name}\n{result_text[:900]}"
                await msg.reply_photo(
                    photo=open(send_path, "rb"),
                    caption=caption[:1024],
                    parse_mode="HTML",
                    read_timeout=60, write_timeout=60, connect_timeout=30,
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
            import base64 as _b64
            rendered_path = post_data.get("rendered_path")
            if rendered_path and Path(rendered_path).exists():
                try:
                    img_b64 = _b64.b64encode(Path(rendered_path).read_bytes()).decode("utf-8")
                    _last_analysis_by_user[user_id] = {"_image_b64": img_b64}
                    logger.info(f"[text] Loaded rendered image as reference for reuse")
                except Exception:
                    pass

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

    logger.info("Bot ready. Polling for Telegram updates...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
