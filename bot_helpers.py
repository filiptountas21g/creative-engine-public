"""Shared helpers + clients used by bot.py and the exec_*.py tool handlers.

Lives outside bot.py so exec_*.py modules can import these without triggering
a circular import (bot.py imports exec_*.py at the dispatcher; exec_*.py
imports back for `brain` and session helpers — without this module, Python's
`__main__` vs module double-execution crashes the container).

Nothing here touches the Telegram Application/Update objects — it's pure state
manipulation + persistence.
"""

import base64
import json
import logging
from dataclasses import asdict
from pathlib import Path

from PIL import Image as PILImage

import config
from brain.client import Brain
from state import (
    MAX_TRACKED_POSTS,
    MAX_VAULT_IMAGES,
    _conversation_memory,
    _image_vault,
    _last_post_by_user,
    _message_counter,
    _posts_by_msg_id,
)

logger = logging.getLogger(__name__)

# ── Shared clients ────────────────────────────────────────
brain = Brain(url=config.TURSO_DATABASE_URL, auth_token=config.TURSO_AUTH_TOKEN)


# ── Image vault ───────────────────────────────────────────
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


# ── Post tracking by Telegram message id ──────────────────
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


# ── Brain-backed post persistence ─────────────────────────
def _persist_user_post(user_id: int):
    """Save the user's last generated post to Brain (survives restarts).
    Stores metadata + template in one row, image files in separate rows."""
    if user_id not in _last_post_by_user:
        return
    post_data = _last_post_by_user[user_id]

    try:
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
                    img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
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


# ── Image compression for Telegram send ───────────────────
def _compress_for_send(img_path: Path) -> Path:
    """Compress PNG → JPEG for faster Telegram uploads. Returns path to send."""
    if img_path.suffix.lower() == ".png":
        with PILImage.open(img_path) as pil_img:
            compressed = img_path.with_suffix(".jpg")
            pil_img.convert("RGB").save(compressed, "JPEG", quality=90)
            orig_kb = img_path.stat().st_size // 1024
            new_kb = compressed.stat().st_size // 1024
            logger.info(f"Compressed PNG→JPEG: {orig_kb}KB → {new_kb}KB")
            return compressed
    return img_path


# ── Sliding conversation memory ───────────────────────────
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
