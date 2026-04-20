"""Per-user in-memory session state.

Module-level dicts holding the bot's runtime state. Lives here (not in bot.py)
so any module can `from state import ...` to share the same state objects
without circular imports.

All entries are mutated in place — never rebound. If you're tempted to write
`_chat_history = {}` somewhere, use `_chat_history.clear()` instead.
"""

# ── Limits / TTLs ─────────────────────────────────────────

MAX_HISTORY = 10  # tight window — prevents corruption from complex tool_use/tool_result pairs
MAX_TRACKED_POSTS = 20  # keep last 20 posts per user in memory
MAX_VAULT_IMAGES = 10
MAX_SAVED_REFERENCES = 5  # keep last 5 references per user (~1MB total)
MEMORY_TTL_MESSAGES = 15  # sliding memory window

# ── Per-user state dicts ──────────────────────────────────

# Last taste analysis per user (for feedback / context in the system prompt)
_last_analysis_by_user: dict[int, dict] = {}

# Full API message history per user — stores tool_use/tool_result blocks
_chat_history: dict[int, list[dict]] = {}

# Last pipeline result per user (for edits, resends)
_last_post_by_user: dict[int, dict] = {}

# Posts indexed by Telegram message_id so the user can reply to any old post
# Key: message_id → post_data dict (same shape as _last_post_by_user values)
_posts_by_msg_id: dict[int, dict] = {}

# Image vault — every image ever used in a post, keyed for fuzzy restore
# Key: user_id → {"hero_<label>": ImageResult, "latest_hero": ImageResult, ...}
_image_vault: dict[int, dict[str, object]] = {}

# Previous CreativeDecisions per user — used to enforce variety across posts
_previous_decisions: dict[int, list[dict]] = {}

# Scout results awaiting user approval
_pending_scout: dict[int, dict] = {}

# Users already restored from Brain this session (prevents double-restore)
_restored_users: set[int] = set()

# Sliding conversation memory
_message_counter: dict[int, int] = {}
_conversation_memory: dict[int, list[dict]] = {}  # list of {"type", "data", "msg_num", "label"}
