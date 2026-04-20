import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── AI Models ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
FAL_KEY = os.getenv("FAL_KEY", "")
IDEOGRAM_API_KEY = os.getenv("IDEOGRAM_API_KEY", "")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

# Model names
OPUS_MODEL = os.getenv("OPUS_MODEL", "claude-opus-4-1-20250805")
SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-4-20250514")
VISION_MODEL = os.getenv("VISION_MODEL", "claude-sonnet-4-20250514")

# ── Telegram ───────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))

# ── Big Brain (Turso) ─────────────────────────────────────
TURSO_DATABASE_URL = os.environ["TURSO_DATABASE_URL"]
TURSO_AUTH_TOKEN = os.environ["TURSO_AUTH_TOKEN"]

# ── Google Drive ───────────────────────────────────────────
DRIVE_INSPIRATION_FOLDER_ID = os.getenv("DRIVE_INSPIRATION_FOLDER_ID", "")
DRIVE_TEMPLATES_FOLDER_ID = os.getenv("DRIVE_TEMPLATES_FOLDER_ID", "120SLEQNFbY8mbw5w1LO2mjmclkGpK_cu")
DRIVE_SCOUT_FOLDER_ID = os.getenv("DRIVE_SCOUT_FOLDER_ID", "")  # Auto-creates subfolder under templates if empty
DRIVE_WATCH_INTERVAL = int(os.getenv("DRIVE_WATCH_INTERVAL", "60"))  # seconds

# ── Rendering ──────────────────────────────────────────────
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Server ─────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "3000"))
