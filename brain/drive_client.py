"""
Google Drive API helpers — adapted for creative-engine (images, not video).

Supports two modes:
  1. Local: Opens browser for OAuth consent → saves token.json
  2. Railway/Cloud: Loads credentials from GOOGLE_TOKEN_JSON and
     GOOGLE_CREDENTIALS_JSON environment variables (no browser needed).
"""

import asyncio
import io
import json
import logging
import os
from pathlib import Path
from functools import lru_cache

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]

_BASE_DIR = Path(__file__).parent.parent  # creative-engine root
_CREDENTIALS_PATH = _BASE_DIR / "credentials.json"
_TOKEN_PATH = _BASE_DIR / "token.json"


def _load_creds_from_env() -> Credentials | None:
    """Try loading Google credentials from environment variables (for Railway)."""
    token_json = os.environ.get("GOOGLE_TOKEN_JSON")
    if not token_json:
        return None
    try:
        token_data = json.loads(token_json)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        logger.info("Loaded Google credentials from GOOGLE_TOKEN_JSON env var")
        return creds
    except Exception as e:
        logger.warning(f"Failed to load credentials from env: {e}")
        return None


def _write_credentials_from_env() -> None:
    """Write credentials.json from env var if it doesn't exist on disk."""
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if creds_json and not _CREDENTIALS_PATH.exists():
        try:
            with open(_CREDENTIALS_PATH, "w") as f:
                f.write(creds_json)
            logger.info("Wrote credentials.json from GOOGLE_CREDENTIALS_JSON env var")
        except Exception as e:
            logger.warning(f"Failed to write credentials.json from env: {e}")


def _save_token(creds: Credentials) -> None:
    """Save refreshed token to disk."""
    try:
        with open(_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        logger.info("Saved refreshed token to token.json")
    except Exception as e:
        logger.warning(f"Could not save token: {e}")


@lru_cache(maxsize=1)
def _get_service():
    """Build and return an authenticated Drive service (cached)."""
    creds = _load_creds_from_env()

    if not creds and _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            _save_token(creds)
        else:
            _write_credentials_from_env()
            if not _CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Google credentials not found at {_CREDENTIALS_PATH}. "
                    "Set GOOGLE_TOKEN_JSON and GOOGLE_CREDENTIALS_JSON env vars."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(_CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)
            _save_token(creds)

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Sync helpers — called from async code via asyncio.to_thread
# ---------------------------------------------------------------------------

def _list_new_images_sync(folder_id: str, seen_ids: set) -> list[dict]:
    """Return file metadata for IMAGE files in folder that are not in seen_ids."""
    svc = _get_service()
    query = (
        f"'{folder_id}' in parents and trashed = false "
        f"and (mimeType contains 'image/')"
    )
    result = svc.files().list(
        q=query, fields="files(id, name, createdTime, mimeType)", orderBy="createdTime desc"
    ).execute()
    files = result.get("files", [])
    return [f for f in files if f["id"] not in seen_ids]


def _download_file_sync(file_id: str, dest_path: Path) -> Path:
    """Download a file from Drive to a local path."""
    svc = _get_service()
    request = svc.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(io.FileIO(dest_path, "wb"), request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    logger.info(f"Downloaded {file_id} → {dest_path}")
    return dest_path


def _get_file_name_sync(file_id: str) -> str:
    """Return the file name for a given Drive file ID."""
    svc = _get_service()
    meta = svc.files().get(fileId=file_id, fields="name").execute()
    return meta.get("name", f"{file_id}.jpg")


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------

async def list_new_images(folder_id: str, seen_ids: set) -> list[dict]:
    return await asyncio.to_thread(_list_new_images_sync, folder_id, seen_ids)


async def download_file(file_id: str, dest_path: Path) -> Path:
    return await asyncio.to_thread(_download_file_sync, file_id, dest_path)


async def get_file_name(file_id: str) -> str:
    return await asyncio.to_thread(_get_file_name_sync, file_id)
