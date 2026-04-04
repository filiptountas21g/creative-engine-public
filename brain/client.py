"""Lectus Big Brain — centralized knowledge store backed by Turso (libSQL) HTTP API."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Optional

import requests


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


def _current_week() -> str:
    return datetime.utcnow().strftime("%G-W%V")


class Brain:
    """Client for reading/writing to the Lectus Big Brain database via Turso HTTP API."""

    def __init__(
        self,
        url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ):
        db_url = url or os.environ.get("TURSO_DATABASE_URL", "")
        self.auth_token = auth_token or os.environ.get("TURSO_AUTH_TOKEN", "")

        if not db_url:
            raise ValueError("TURSO_DATABASE_URL not set")

        # Convert libsql:// URL to HTTPS for HTTP API
        self.http_url = db_url.replace("libsql://", "https://")
        if not self.http_url.startswith("https://"):
            self.http_url = f"https://{self.http_url}"

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        })

    def _execute(self, sql: str, args: list = None) -> list:
        """Execute a SQL statement via Turso HTTP API v2 (pipeline)."""
        # Build the request body
        stmt = {"type": "execute", "stmt": {"sql": sql}}
        if args:
            stmt["stmt"]["args"] = [
                {"type": "null", "value": None} if v is None
                else {"type": "integer", "value": str(v)} if isinstance(v, int)
                else {"type": "text", "value": str(v)}
                for v in args
            ]

        body = {
            "requests": [
                stmt,
                {"type": "close"},
            ]
        }

        # Retry up to 3 times on connection errors
        import time as _time
        last_err = None
        for attempt in range(3):
            try:
                resp = self.session.post(f"{self.http_url}/v2/pipeline", json=body, timeout=15)
                resp.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < 2:
                    _time.sleep(1)
                    continue
                raise last_err
        data = resp.json()

        # Extract results from pipeline response
        results = data.get("results", [])
        if not results:
            return []

        result = results[0]
        if result.get("type") == "error":
            raise Exception(f"Turso error: {result['error']['message']}")

        response = result.get("response", {})
        if response.get("type") != "execute":
            return []

        execute_result = response.get("result", {})
        rows = execute_result.get("rows", [])
        return [[col.get("value") for col in row] for row in rows]

    # ── WRITE ────────────────────────────────────────────────────────

    def store(
        self,
        topic: str,
        source: str,
        content: str,
        client: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[list] = None,
        week: Optional[str] = None,
    ) -> None:
        """Store a knowledge entry."""
        tags_str = ",".join(tags) if tags else None
        week = week or _current_week()

        self._execute(
            """INSERT INTO brain_entries (client, topic, source, content, summary, tags, week, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [client or "ALL", topic, source, content, summary, tags_str, week, _now()],
        )

    def delete_by_topic_source(self, topic: str, source: str) -> None:
        """Delete all entries matching a topic + source combo.
        Used to keep session data (1 row per user) from accumulating."""
        self._execute(
            "DELETE FROM brain_entries WHERE topic = ? AND source = ?",
            [topic, source],
        )

    def store_batch(self, entries: list) -> int:
        """Store multiple entries. Returns count stored."""
        count = 0
        for entry in entries:
            self.store(
                topic=entry["topic"],
                source=entry["source"],
                content=entry["content"],
                client=entry.get("client"),
                summary=entry.get("summary"),
                tags=entry.get("tags"),
                week=entry.get("week"),
            )
            count += 1
        return count

    # ── READ ─────────────────────────────────────────────────────────

    def _rows_to_dicts(self, rows: list) -> list:
        """Convert raw rows to dicts."""
        return [
            {
                "id": r[0],
                "client": r[1],
                "topic": r[2],
                "source": r[3],
                "content": r[4],
                "summary": r[5],
                "tags": r[6].split(",") if r[6] else [],
                "week": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]

    def query(
        self,
        client: Optional[str] = None,
        topic: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[list] = None,
        since: Optional[str] = None,
        week: Optional[str] = None,
        limit: int = 20,
    ) -> list:
        """Query brain entries with filters."""
        conditions = []
        params = []

        if client:
            conditions.append("(LOWER(client) = LOWER(?) OR client = 'ALL')")
            params.append(client)
        if topic:
            conditions.append("topic = ?")
            params.append(topic)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if week:
            conditions.append("week = ?")
            params.append(week)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        rows = self._execute(
            f"""SELECT id, client, topic, source, content, summary, tags, week, created_at
                FROM brain_entries
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ?""",
            params,
        )
        return self._rows_to_dicts(rows)

    def get_client_context(self, client_name: str, weeks: int = 4) -> list:
        """Get all recent knowledge about a specific client (last N weeks)."""
        since = (datetime.utcnow() - timedelta(weeks=weeks)).strftime("%Y-%m-%dT%H:%M:%S")
        return self.query(client=client_name, since=since, limit=50)

    def get_industry_trends(self, industry: str, limit: int = 10) -> list:
        """Get recent trend data matching an industry keyword."""
        rows = self._execute(
            """SELECT id, client, topic, source, content, summary, tags, week, created_at
               FROM brain_entries
               WHERE (topic IN ('content_trends', 'instagram_trends', 'creative_ideas'))
                 AND (content LIKE ? OR tags LIKE ?)
               ORDER BY created_at DESC
               LIMIT ?""",
            [f"%{industry}%", f"%{industry}%", limit],
        )
        return self._rows_to_dicts(rows)

    def search(self, query: str, limit: int = 20) -> list:
        """Full-text search across all brain entries."""
        rows = self._execute(
            """SELECT id, client, topic, source, content, summary, tags, week, created_at
               FROM brain_entries
               WHERE content LIKE ? OR summary LIKE ? OR tags LIKE ? OR client LIKE ?
               ORDER BY created_at DESC
               LIMIT ?""",
            [f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit],
        )
        return self._rows_to_dicts(rows)

    def get_latest_by_topic(self, topic: str) -> list:
        """Get the most recent entry for each client for a given topic."""
        rows = self._execute(
            """SELECT id, client, topic, source, content, summary, tags, week, created_at
               FROM brain_entries
               WHERE topic = ?
               GROUP BY client
               HAVING created_at = MAX(created_at)
               ORDER BY client""",
            [topic],
        )
        return self._rows_to_dicts(rows)

    # ── CLIENTS ──────────────────────────────────────────────────────

    def get_clients(self, active_only: bool = True) -> list:
        """Get all clients from the brain."""
        where = "WHERE active = 1" if active_only else ""
        rows = self._execute(
            f"SELECT name, industry, website, description, products, platforms, tone, target_audience FROM clients {where} ORDER BY name",
        )
        return [
            {
                "name": r[0],
                "industry": r[1],
                "website": r[2],
                "description": r[3],
                "products": json.loads(r[4]) if r[4] else [],
                "platforms": json.loads(r[5]) if r[5] else [],
                "tone": r[6],
                "target_audience": r[7],
            }
            for r in rows
        ]

    def upsert_client(self, name: str, **kwargs) -> None:
        """Insert or update a client."""
        if "products" in kwargs and isinstance(kwargs["products"], list):
            kwargs["products"] = json.dumps(kwargs["products"])
        if "platforms" in kwargs and isinstance(kwargs["platforms"], list):
            kwargs["platforms"] = json.dumps(kwargs["platforms"])

        existing = self._execute("SELECT id FROM clients WHERE name = ?", [name])

        if existing:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [name]
            self._execute(f"UPDATE clients SET {sets} WHERE name = ?", vals)
        else:
            cols = ["name"] + list(kwargs.keys())
            placeholders = ", ".join("?" for _ in cols)
            vals = [name] + list(kwargs.values())
            self._execute(
                f"INSERT INTO clients ({', '.join(cols)}) VALUES ({placeholders})", vals
            )

    # ── AGENT MEMORY ─────────────────────────────────────────────────

    def get_memory(self, agent: str, key: str):
        """Read agent memory by key."""
        rows = self._execute(
            "SELECT value FROM agent_memory WHERE agent = ? AND key = ? ORDER BY updated_at DESC LIMIT 1",
            [agent, key],
        )
        if rows:
            return json.loads(rows[0][0])
        return None

    def set_memory(self, agent: str, key: str, value: dict) -> None:
        """Write agent memory (upsert)."""
        existing = self._execute(
            "SELECT id FROM agent_memory WHERE agent = ? AND key = ?",
            [agent, key],
        )

        json_val = json.dumps(value, ensure_ascii=False)
        if existing:
            self._execute(
                "UPDATE agent_memory SET value = ?, updated_at = ? WHERE agent = ? AND key = ?",
                [json_val, _now(), agent, key],
            )
        else:
            self._execute(
                "INSERT INTO agent_memory (agent, key, value, updated_at) VALUES (?, ?, ?, ?)",
                [agent, key, json_val, _now()],
            )

    # ── STATS ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get brain statistics."""
        total = self._execute("SELECT COUNT(*) FROM brain_entries")[0][0]
        by_topic = self._execute(
            "SELECT topic, COUNT(*) FROM brain_entries GROUP BY topic ORDER BY COUNT(*) DESC"
        )
        by_client = self._execute(
            "SELECT client, COUNT(*) FROM brain_entries GROUP BY client ORDER BY COUNT(*) DESC"
        )
        clients = self._execute("SELECT COUNT(*) FROM clients WHERE active = 1")[0][0]

        return {
            "total_entries": int(total) if total else 0,
            "active_clients": int(clients) if clients else 0,
            "by_topic": {r[0]: int(r[1]) for r in by_topic},
            "by_client": {r[0]: int(r[1]) for r in by_client},
        }
