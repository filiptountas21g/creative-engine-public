"""Design Scout — Perplexity finds fresh layout inspiration, Claude extracts actionable blueprints.

Two modes:
1. Manual — user says "find me fresh designs" → runs scout, shows results
2. Automatic — anti-repetition detects staleness → scout runs silently, feeds Opus

Layout tags are stored per post in Brain for staleness detection.

When user picks a scout result ("make a post like number 3"), the system:
1. Downloads the actual design image from the URL
2. Runs full Vision analysis (same as inspiration photos)
3. Passes it as forced_reference — Opus SEES the layout + gets "CLOSELY MATCH" instructions
4. Critique checks the render against the reference image
"""

import base64
import json
import logging
import random
import tempfile
from pathlib import Path
from typing import Optional

import httpx
import anthropic

import config
from brain.client import Brain

logger = logging.getLogger(__name__)

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Layout tag categories for anti-repetition tracking
LAYOUT_CATEGORIES = {
    "headline_position": [
        "top-left", "top-center", "top-right",
        "center-left", "centered", "center-right",
        "bottom-left", "bottom-center", "bottom-right",
        "overlapping-image",
    ],
    "image_layout": [
        "full-bleed", "split-left", "split-right", "split-top", "split-bottom",
        "grid", "circular-crop", "overlapping", "diagonal", "floating",
        "multi-photo", "no-image", "background-only",
    ],
    "composition": [
        "symmetric", "asymmetric", "editorial", "minimal", "maximalist",
        "collage", "broken-grid", "layered", "single-column", "magazine",
    ],
    "text_weight": [
        "text-heavy", "balanced", "image-dominant", "text-only",
    ],
}


# ── Layout Tag Extraction (runs during critique, zero extra cost) ──────

LAYOUT_TAG_PROMPT = """Look at this rendered social media post and classify its layout.

Return ONLY valid JSON with these 4 fields:
{
    "headline_position": one of: "top-left", "top-center", "top-right", "center-left", "centered", "center-right", "bottom-left", "bottom-center", "bottom-right", "overlapping-image",
    "image_layout": one of: "full-bleed", "split-left", "split-right", "split-top", "split-bottom", "grid", "circular-crop", "overlapping", "diagonal", "floating", "multi-photo", "no-image", "background-only",
    "composition": one of: "symmetric", "asymmetric", "editorial", "minimal", "maximalist", "collage", "broken-grid", "layered", "single-column", "magazine",
    "text_weight": one of: "text-heavy", "balanced", "image-dominant", "text-only"
}

Return ONLY the JSON, nothing else."""


async def extract_layout_tags(image_path: str) -> dict:
    """Extract layout classification tags from a rendered post image.
    Uses Haiku for speed and cost (~$0.001 per call)."""
    import base64
    from pathlib import Path

    try:
        img_bytes = Path(image_path).read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        suffix = Path(image_path).suffix.lower()
        media = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-20250414",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": img_b64}},
                    {"type": "text", "text": LAYOUT_TAG_PROMPT},
                ],
            }],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rstrip("`").strip()
        tags = json.loads(raw)
        logger.info(f"Layout tags: {tags}")
        return tags

    except Exception as e:
        logger.error(f"Layout tag extraction failed: {e}")
        return {}


def store_layout_tags(brain: Brain, client: str, tags: dict, headline: str = ""):
    """Store layout tags in Brain for anti-repetition tracking."""
    if not tags:
        return
    brain.store(
        topic="layout_tags",
        source="pipeline",
        content=json.dumps(tags, ensure_ascii=False),
        client=client,
        summary=f"{tags.get('composition', '?')} / {tags.get('headline_position', '?')} / {tags.get('image_layout', '?')}",
        tags=list(tags.values()) + ["layout_tracking"],
    )


def detect_staleness(brain: Brain, client: str = "ALL", lookback: int = 6) -> dict:
    """Check recent posts for layout repetition. Returns staleness info.

    Returns:
        {
            "is_stale": bool,
            "repeated_patterns": {"headline_position": "centered (4/6)", ...},
            "avoid_instructions": "Avoid centered headline, full-bleed image, symmetric composition.",
            "should_scout": bool,  # True if local references can't help
        }
    """
    # Get recent layout tags
    recent = brain.query(topic="layout_tags", client=client, limit=lookback)
    if not recent:
        # Also check ALL
        recent = brain.query(topic="layout_tags", limit=lookback)

    if len(recent) < 3:
        return {"is_stale": False, "repeated_patterns": {}, "avoid_instructions": "", "should_scout": False}

    # Count frequency of each tag value
    tag_counts = {}
    for entry in recent:
        try:
            tags = json.loads(entry["content"])
            for key, val in tags.items():
                if key not in tag_counts:
                    tag_counts[key] = {}
                tag_counts[key][val] = tag_counts[key].get(val, 0) + 1
        except (json.JSONDecodeError, KeyError):
            continue

    total = len(recent)
    threshold = 0.5  # If same value appears in >50% of recent posts, it's stale
    repeated = {}
    avoid_parts = []

    for category, counts in tag_counts.items():
        for value, count in counts.items():
            ratio = count / total
            if ratio >= threshold:
                repeated[category] = f"{value} ({count}/{total})"
                avoid_parts.append(f"{value} {category.replace('_', ' ')}")

    is_stale = len(repeated) >= 2  # At least 2 categories are repetitive
    should_scout = len(repeated) >= 3  # Very stale — need external inspiration

    avoid_instructions = ""
    if avoid_parts:
        avoid_instructions = "AVOID these layout patterns (overused recently): " + ", ".join(avoid_parts) + ". Try something structurally different."

    return {
        "is_stale": is_stale,
        "repeated_patterns": repeated,
        "avoid_instructions": avoid_instructions,
        "should_scout": should_scout,
    }


# ── Perplexity Design Scout ───────────────────────────────────────


def _get_seen_urls(brain: Brain) -> set:
    """Get URLs already found in previous scout runs from Brain."""
    seen = set()
    try:
        entries = brain.query(topic="design_scout", limit=20)
        for entry in entries:
            content = entry.get("content", "")
            # Extract URLs from stored content
            import re
            urls = re.findall(r'https?://[^\s\)\"\'<>]+', content)
            seen.update(urls)
        # Also check scout_seen_urls topic
        url_entries = brain.query(topic="scout_seen_urls", limit=5)
        for entry in url_entries:
            try:
                urls = json.loads(entry.get("content", "[]"))
                seen.update(urls)
            except (json.JSONDecodeError, TypeError):
                pass
    except Exception:
        pass
    return seen


def _recency_weight(created_at: str) -> float:
    """Return a weight multiplier based on how recently an entry was created.
    Recent entries count more — taste evolves over time.
    0-7 days: 1.0, 8-30 days: 0.6, 31-90 days: 0.3, 90+ days: 0.1
    """
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).days
        if age_days <= 7:
            return 1.0
        elif age_days <= 30:
            return 0.6
        elif age_days <= 90:
            return 0.3
        else:
            return 0.1
    except Exception:
        return 0.5  # Unknown age → neutral weight


def _get_taste_description(brain: Brain, client: str = None) -> str:
    """Get the user's taste profile as natural language, with recency weighting.

    Recent approvals count more — if you've been into minimalism lately,
    that shows up stronger than editorial layouts you liked 3 months ago.

    If client is provided, also pulls client-specific brand preferences.
    """
    try:
        refs = brain.query(topic="taste_reference", limit=50)
        if not refs:
            return ""

        # Extract patterns with recency weighting
        from collections import defaultdict
        fonts: dict = defaultdict(float)
        compositions: dict = defaultdict(float)
        moods: dict = defaultdict(float)
        colors: dict = defaultdict(float)
        rules: dict = defaultdict(float)

        for ref in refs:
            try:
                weight = _recency_weight(ref.get("created_at", ""))
                data = json.loads(ref.get("content", "{}"))

                # Typography — list of dicts
                typo = data.get("typography", [])
                if isinstance(typo, list):
                    for elem in typo:
                        cat = elem.get("font_category", "")
                        if cat:
                            fonts[cat] += weight
                elif isinstance(typo, dict):
                    cat = typo.get("font_category", "")
                    if cat:
                        fonts[cat] += weight

                # Composition
                comp = data.get("composition", {})
                if isinstance(comp, dict) and comp.get("template_match"):
                    compositions[comp["template_match"]] += weight

                # Feeling / mood
                feeling = data.get("feeling", {})
                if isinstance(feeling, dict) and feeling.get("mood"):
                    moods[feeling["mood"]] += weight

                # Colors
                color_data = data.get("colors", {})
                if isinstance(color_data, dict):
                    temp = color_data.get("temperature", "")
                    if temp:
                        colors[temp] += weight
                    palette_mood = color_data.get("palette_mood", "")
                    if palette_mood:
                        colors[palette_mood] += weight

                # Reusable rules
                for rule in data.get("reusable_rules", []):
                    if isinstance(rule, str) and len(rule) > 10:
                        rules[rule] += weight

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        # Sort by weighted score (most recent/frequent first)
        def top(d: dict, n: int) -> list:
            return [k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:n]]

        parts = []
        if fonts:
            parts.append(f"Typography: {', '.join(top(fonts, 3))}")
        if compositions:
            parts.append(f"Layouts: {', '.join(top(compositions, 3))}")
        if moods:
            parts.append(f"Mood: {', '.join(top(moods, 3))}")
        if colors:
            parts.append(f"Colors: {', '.join(top(colors, 3))}")
        if rules:
            parts.append(f"Design rules: {'; '.join(top(rules, 3))}")

        taste = "; ".join(parts) if parts else ""

        # In client mode — also pull client brand context
        if client and client != "ALL":
            client_ctx = _get_client_brand_context(brain, client)
            if client_ctx:
                taste = f"{taste}\n\nClient brand context ({client}): {client_ctx}"

        return taste

    except Exception:
        return ""


def _get_client_brand_context(brain: Brain, client: str) -> str:
    """Pull a client's brand profile from Brain for client-mode scouting."""
    try:
        entries = brain.query(topic="client_preference", client=client, limit=5)
        if not entries:
            entries = brain.query(topic="brand_logo", client=client, limit=3)
        if not entries:
            return ""

        parts = []
        for entry in entries[:3]:
            content = entry.get("content", "")
            summary = entry.get("summary", "")
            if summary:
                parts.append(summary)
            elif content:
                parts.append(content[:200])

        return " | ".join(parts) if parts else ""
    except Exception:
        return ""
    except Exception:
        return ""


def _build_smart_query(
    brain: Brain,
    industry: str,
    staleness: dict = None,
    seen_urls: set = None,
    user_focus: str = "",
) -> str:
    """Use Claude to build a targeted search query from taste profile + context.
    Like Poulxeria does — not hardcoded generic queries."""

    taste_text = _get_taste_description(brain)

    avoid_hint = ""
    if staleness and staleness.get("repeated_patterns"):
        patterns = [v.split(" (")[0] for v in staleness["repeated_patterns"].values()]
        avoid_hint = f"Recently overused: {', '.join(patterns)}. Find something structurally different."

    exclude_hint = ""
    if seen_urls:
        import re
        domains = set()
        for url in list(seen_urls)[:15]:
            m = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if m:
                domains.add(m.group(1))
        if domains:
            exclude_hint = f"Already seen sources (avoid these): {', '.join(list(domains)[:8])}"

    focus_text = f"User specifically wants: {user_focus}" if user_focus else "Find fresh, surprising inspiration"

    _ai = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = _ai.messages.create(
        model=config.SONNET_MODEL,
        max_tokens=400,
        system=(
            "You write search queries that find REAL social media posts and design work. "
            "Not articles about design trends. Not template marketplaces. Not font lists. "
            "ACTUAL posts from real brands, agency portfolios on Behance/Dribbble, "
            "Pinterest pins of ad creatives, Instagram brand accounts.\n\n"
            "Return ONLY the search query text — nothing else. No quotes, no explanation. "
            "Make it specific and targeted based on the taste profile and context."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Industry: {industry}\n"
                f"Taste profile: {taste_text}\n"
                f"{avoid_hint}\n"
                f"{exclude_hint}\n"
                f"{focus_text}\n\n"
                f"Write a search query to find 5-7 REAL social media posts and design work "
                f"that would inspire this user. Prioritize 2024-2026 work."
            ),
        }],
    )

    return response.content[0].text.strip()


def _build_scout_queries(brain: Brain, industry: str, staleness: dict = None, seen_urls: set = None, user_focus: str = "", client: str = None) -> list[tuple[str, str]]:
    """Build 3 targeted search queries using Claude — different every time.

    In client mode (client != None/'ALL'), blends taste with client brand context.
    In global mode, purely taste-driven.
    """
    # Pass client to get blended taste + brand context if in client mode
    taste_text = _get_taste_description(brain, client=client if client and client != "ALL" else None)

    avoid_hint = ""
    if staleness and staleness.get("repeated_patterns"):
        patterns = [v.split(" (")[0] for v in staleness["repeated_patterns"].values()]
        avoid_hint = f"Recently overused layouts: {', '.join(patterns)}. Find something structurally different."

    exclude_hint = ""
    if seen_urls:
        import re
        domains = set()
        for url in list(seen_urls)[:15]:
            m = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if m:
                domains.add(m.group(1))
        if domains:
            exclude_hint = f"Already seen (skip these domains): {', '.join(list(domains)[:8])}"

    focus_text = f"User specifically wants: {user_focus}" if user_focus else ""

    _ai = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = _ai.messages.create(
        model=config.SONNET_MODEL,
        max_tokens=600,
        system=(
            "You write Google Image search queries that find FINISHED SOCIAL MEDIA POST DESIGNS "
            "— the actual square/portrait graphics a brand posts on Instagram or LinkedIn.\n\n"
            "CRITICAL: We want FINISHED POSTS, not:\n"
            "  ✗ Font specimen sheets or typeface showcases\n"
            "  ✗ Logo design or brand identity work\n"
            "  ✗ Website or app UI screenshots\n"
            "  ✗ Design articles or tutorials\n"
            "  ✗ Template marketplace listings\n"
            "  ✗ Typography posters\n\n"
            "We want:\n"
            "  ✓ Instagram post designs with headline + image + brand name\n"
            "  ✓ Social media graphics showing a product or concept\n"
            "  ✓ Agency-made brand campaign posts\n"
            "  ✓ Square or portrait format marketing creatives\n\n"
            "GOOD QUERY PATTERNS:\n"
            "  site:behance.net social media posts campaign brand {industry} 2025\n"
            "  site:dribbble.com instagram post design brand {mood} layout\n"
            "  site:pinterest.com instagram post graphic design brand campaign minimal\n\n"
            "ALWAYS include 'social media post' OR 'instagram post' OR 'brand campaign post' in each query.\n"
            "NEVER use just typography or font terms without 'social media post' context.\n"
            "Use site: filters for at least 2 of the 3 queries.\n\n"
            "Return EXACTLY 3 queries, one per line. No numbering, no quotes, no explanation."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"{'CLIENT MODE: ' + client if client and client != 'ALL' else 'GLOBAL MODE (personal taste only)'}\n"
                f"Industry: {industry}\n"
                f"Designer taste (use for mood/color/composition direction, NOT font terms):\n{taste_text}\n"
                f"{avoid_hint}\n"
                f"{exclude_hint}\n"
                f"{focus_text}\n\n"
                f"Write 3 image search queries to find FINISHED SOCIAL MEDIA POST DESIGNS. "
                f"Each query MUST target actual posts (Instagram/brand campaign graphics), not fonts or logos. "
                f"Each from a different source (Behance, Dribbble/Pinterest, open web). "
                f"Prioritize 2024-2026 work."
            ),
        }],
    )

    raw = response.content[0].text.strip()
    lines = [line.strip() for line in raw.split("\n") if line.strip() and not line.strip().startswith(("#", "//"))]

    # Clean up any numbering
    import re
    cleaned = []
    for line in lines[:3]:
        line = re.sub(r'^[\d]+[\.\)\-]\s*', '', line).strip().strip('"').strip("'")
        if len(line) > 20:
            cleaned.append(line)

    labels = [f"Industry: {industry}", "Portfolio scouting", "Experimental"]
    queries = []
    for i, q in enumerate(cleaned):
        label = labels[i] if i < len(labels) else f"Search {i+1}"
        queries.append((q, label))

    # Fallback if Claude gave less than 3
    if len(queries) < 2:
        queries.append((
            f"site:behance.net instagram post design brand campaign {industry} social media graphic 2025",
            f"Industry: {industry}",
        ))
    if len(queries) < 3:
        queries.append((
            "site:pinterest.com instagram post design brand campaign minimal editorial social media graphic",
            "Experimental",
        ))

    return queries


# Domains we accept as design inspiration sources
GOOD_DESIGN_DOMAINS = {
    "pinterest.com", "behance.net", "dribbble.com", "instagram.com",
    "awwwards.com", "mindsparklemag.com", "designinspiration.com",
    "itsnicethat.com", "designmilk.com", "visuelle.co.uk",
    "abduzeedo.com", "designspiration.com", "fonts.google.com",
    "typewolf.com", "brandnew.com", "underconsideration.com",
}

# Domains we always block
HARD_BLOCKED_DOMAINS = {
    "canva.com", "envato.com", "freepik.com", "shutterstock.com",
    "stock.adobe.com", "creativemarket.com", "templatemonster.com",
    "vecteezy.com", "pngtree.com", "slidesgo.com", "placeit.net",
}


async def _serper_image_search(query: str, num: int = 10) -> list[dict]:
    """Search Google Images via Serper — returns direct image URLs from Pinterest/Behance/Dribbble.

    Returns: [{"imageUrl", "link", "title", "domain", "imageWidth", "imageHeight"}]
    """
    if not config.SERPER_API_KEY:
        logger.warning("No SERPER_API_KEY set")
        return []

    try:
        async with httpx.AsyncClient(timeout=30) as http:
            response = await http.post(
                "https://google.serper.dev/images",
                headers={
                    "X-API-KEY": config.SERPER_API_KEY,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num},
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("images", [])
            logger.info(f"Serper returned {len(results)} images for: {query[:60]}")
            return results
    except Exception as e:
        logger.error(f"Serper image search failed: {e}")
        return []


def _is_good_design_image(item: dict, seen_image_urls: set) -> bool:
    """Filter: keep only real design images from quality sources."""
    import re
    domain = item.get("domain", "")
    image_url = item.get("imageUrl", "")
    w = item.get("imageWidth", 0)
    h = item.get("imageHeight", 0)

    # Must have an image URL
    if not image_url or not image_url.startswith("http"):
        return False

    # Skip already seen
    if image_url in seen_image_urls:
        return False

    # Skip hard-blocked domains
    if any(blocked in domain for blocked in HARD_BLOCKED_DOMAINS):
        return False

    # Skip tiny images (thumbnails, icons)
    if w and h and (w < 200 or h < 200):
        return False

    # Skip obvious non-design images
    lower_url = image_url.lower()
    if any(ext in lower_url for ext in [".gif", ".svg", ".ico", ".webp?w=50"]):
        return False

    return True


async def scout_search(
    brain: Brain,
    client: str = "ALL",
    industry: str = None,
    staleness: dict = None,
    user_focus: str = "",
) -> dict:
    """Phase 1: Search Google Images via Serper — returns real design images from
    Pinterest, Behance, Dribbble for user approval. Does NOT store anything yet.

    Returns:
        {
            "items": [{"index", "name", "description", "url", "image_url", "domain"}],
            "raw_images": [...],   # Full Serper results, kept for phase 2
            "client": str,
            "industry": str,
            "staleness": dict,
        }
    """
    import asyncio, re as _re

    # Determine industry from Brain if not provided
    if not industry:
        try:
            clients = brain.get_clients()
            for c in clients:
                if c["name"].lower() == client.lower():
                    industry = c.get("industry", "business")
                    break
            if not industry:
                industry = "business and design"
        except Exception:
            industry = "business and design"

    logger.info(f"Design scout SEARCH (Serper) for {client} ({industry})")

    # Get previously seen image URLs to avoid duplicates
    seen_urls = _get_seen_urls(brain)
    seen_image_urls = set()
    for entry in brain.query(topic="scout_seen_urls", limit=10):
        try:
            seen_image_urls.update(json.loads(entry.get("content", "[]")))
        except Exception:
            pass

    # Claude builds queries optimised for Google Image search
    queries = _build_scout_queries(brain, industry, staleness, seen_urls, user_focus, client=client)

    # Run all Serper image searches in parallel
    tasks = [_serper_image_search(q, num=15) for q, _ in queries]
    search_batches = await asyncio.gather(*tasks)

    # Combine + filter all image results
    all_images = []
    seen_img_dedup = set()
    for batch in search_batches:
        for img in batch:
            url = img.get("imageUrl", "")
            if url and url not in seen_img_dedup and _is_good_design_image(img, seen_image_urls):
                seen_img_dedup.add(url)
                all_images.append(img)

    if not all_images:
        logger.warning("Serper returned no usable design images")
        return {"items": [], "raw_images": [], "client": client, "industry": industry, "staleness": staleness}

    logger.info(f"Scout: {len(all_images)} unique images after filtering")

    # Get taste for Claude curation
    taste_text = _get_taste_description(brain)

    # Build summary of what we found for Claude to curate
    images_summary = "\n".join(
        f'{i+1}. [{img.get("domain","?")}] {img.get("title","")[:80]} | {img.get("link","")[:80]}'
        for i, img in enumerate(all_images[:25])
    )

    # Claude picks the best 15 — showing more so user can teach the system their taste
    _ai = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = _ai.messages.create(
        model=config.SONNET_MODEL,
        max_tokens=2000,
        system=(
            "You are a senior art director curating SOCIAL MEDIA POST designs for inspiration.\n"
            "You are looking for FINISHED POSTS — the kind a brand actually publishes on Instagram or LinkedIn.\n\n"
            "INCLUDE: brand campaign posts, product posts, editorial graphics, lifestyle posts, "
            "announcement posts — anything that looks like a real published social media graphic.\n\n"
            "EXCLUDE (skip these entirely):\n"
            "  - Font specimen sheets or typeface showcases (stacked words showing font weights)\n"
            "  - Logo design or brand identity work\n"
            "  - Website screenshots or UI/app mockups\n"
            "  - Design tutorials or how-to guides\n"
            "  - Template marketplace previews\n"
            "  - YouTube thumbnails or video covers\n\n"
            "Prefer images from pinterest.com, behance.net, dribbble.com.\n"
            "Return ONLY valid JSON."
        ),
        messages=[{
            "role": "user",
            "content": f"""My taste profile (recent preferences weighted higher):
{taste_text}

Here are images found via search. I need SOCIAL MEDIA POST designs only.
Pick up to 15 that are actual finished social media posts (not fonts, not logos, not UI).
Include a range of layouts and styles so I can react and refine taste.

IMAGES FOUND:
{images_summary}

Return JSON array:
[
  {{
    "index": <number from the list above>,
    "name": "Short name 5 words max",
    "description": "Layout, composition, mood — what the post actually looks like",
    "why_relevant": "Why this is useful as social media post inspiration"
  }},
  ...
]

SKIP anything that looks like a font showcase, logo, UI screenshot, or tutorial.
Return ONLY the JSON array."""
        }],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1].rstrip("`").strip()

    # Parse Claude's picks and map back to actual image data
    items = []
    try:
        picks = json.loads(raw_text)
        if isinstance(picks, list):
            for pick in picks:
                idx = pick.get("index", 0) - 1  # Convert to 0-based
                if 0 <= idx < len(all_images):
                    img = all_images[idx]
                    items.append({
                        "index": len(items) + 1,
                        "name": pick.get("name", img.get("title", "Design")[:40]),
                        "description": pick.get("description", ""),
                        "why_relevant": pick.get("why_relevant", ""),
                        "url": img.get("link", ""),
                        "image_url": img.get("imageUrl", ""),
                        "thumbnail_url": img.get("thumbnailUrl", ""),
                        "domain": img.get("domain", ""),
                    })
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Claude curation parse failed: {e}, using top images directly")
        # Fallback: just take top 15 images
        for i, img in enumerate(all_images[:15]):
            items.append({
                "index": i + 1,
                "name": img.get("title", "Design")[:40],
                "description": f"Design from {img.get('domain', '?')}",
                "why_relevant": "",
                "url": img.get("link", ""),
                "image_url": img.get("imageUrl", ""),
                "thumbnail_url": img.get("thumbnailUrl", ""),
                "domain": img.get("domain", ""),
            })

    logger.info(f"Scout search found {len(items)} curated items from Serper")

    return {
        "items": items,
        "raw_images": all_images,  # Keep for phase 2
        "client": client,
        "industry": industry or "",
        "staleness": staleness,
    }


async def scout_approve(
    brain: Brain,
    pending: dict,
    approved_indices: list[int],
    staleness: dict = None,
) -> dict:
    """Phase 2: User approved specific items — now extract full blueprints and store.

    Args:
        brain: Brain instance
        pending: The dict returned by scout_search()
        approved_indices: Which items (1-based) the user approved
        staleness: Optional staleness context

    Returns:
        {"layouts_found": int, "layouts_text": str, "stored": bool}
    """
    items = pending.get("items", [])
    client = pending.get("client", "ALL")
    staleness = staleness or pending.get("staleness")

    # Build list of approved items
    approved_items = [item for item in items if item["index"] in approved_indices]
    if not approved_items:
        return {"layouts_found": 0, "layouts_text": "No items selected.", "stored": False}

    logger.info(f"Scout approve: storing {len(approved_items)} approved items")

    # Store a brief summary of approved designs in Brain for future reference
    approved_summary = "\n".join(
        f"- {item.get('name', '?')} ({item.get('domain', '?')}): {item.get('description', '')}"
        for item in approved_items
    )
    brain.store(
        topic="design_scout",
        source="serper",
        content=approved_summary,
        client=client,
        summary=f"Approved scout designs ({len(approved_items)} items)",
        tags=["design_scout", "layout_inspiration", "approved"],
    )

    # Store all image URLs from this scout run so they don't repeat next time
    all_urls = []
    for item in items:
        if item.get("url"):
            all_urls.append(item["url"])
        if item.get("image_url"):
            all_urls.append(item["image_url"])
    all_urls = list(set(u for u in all_urls if u))

    if all_urls:
        brain.store(
            topic="scout_seen_urls",
            source="scout",
            content=json.dumps(all_urls),
            client=client,
            summary=f"Scout URLs batch ({len(all_urls)} URLs)",
            tags=["scout_seen_urls"],
        )

    logger.info(f"Scout: stored {len(approved_items)} approved layouts + {len(all_urls)} seen URLs in Brain")

    return {
        "layouts_found": len(approved_names),
        "layouts_text": layouts_text,
        "stored": True,
    }


async def run_design_scout_auto(
    brain: Brain,
    client: str = "ALL",
    industry: str = None,
    staleness: dict = None,
) -> dict:
    """Automatic scout (no user approval) — light pass only.
    Stores raw text descriptions as hints, no deep Vision analysis.
    Used by the anti-repetition system when it detects staleness."""

    if not industry:
        try:
            clients = brain.get_clients()
            for c in clients:
                if c["name"].lower() == client.lower():
                    industry = c.get("industry", "business")
                    break
            if not industry:
                industry = "business and design"
        except Exception:
            industry = "business and design"

    logger.info(f"Auto scout (light pass, Serper) for {client} ({industry})")

    import asyncio
    queries = _build_scout_queries(brain, industry, staleness)
    tasks = [_serper_image_search(q, num=8) for q, _ in queries]
    batches = await asyncio.gather(*tasks)

    # Combine + filter images
    all_images = []
    seen_set = set()
    for batch in batches:
        for img in batch:
            url = img.get("imageUrl", "")
            if url and url not in seen_set and _is_good_design_image(img, set()):
                seen_set.add(url)
                all_images.append(img)

    if not all_images:
        return {"layouts_found": 0, "stored": False}

    # Store lightweight summary (just titles + domains, no Vision analysis)
    layouts_text = "\n".join(
        f"- [{img.get('domain','?')}] {img.get('title','')[:80]} | {img.get('link','')[:80]}"
        for img in all_images[:10]
    )

    brain.store(
        topic="design_scout",
        source="serper_auto",
        content=layouts_text,
        client=client,
        summary=f"Auto-scout layout hints ({len(valid_results)} sources)",
        tags=["design_scout", "auto", "layout_hints"],
    )

    layout_count = layouts_text.count("**Name**") or layouts_text.count("##")
    if layout_count == 0:
        layout_count = len([line for line in layouts_text.split("\n") if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8."))])

    return {"layouts_found": layout_count, "stored": True}


def get_scout_inspiration(brain: Brain, client: str = "ALL", staleness: dict = None) -> Optional[str]:
    """Pull a random layout blueprint from stored scout data.
    Used during generation to feed Opus a fresh structural constraint."""

    entries = brain.query(topic="design_scout", client=client, limit=5)
    if not entries:
        entries = brain.query(topic="design_scout", limit=5)

    if not entries:
        return None

    # Pick a random entry
    entry = random.choice(entries)
    full_text = entry["content"]

    # Try to pick a single random layout from the text
    # Split by numbered headers or ## headers
    import re
    sections = re.split(r'\n(?=(?:\d+\.|##\s))', full_text)
    sections = [s.strip() for s in sections if len(s.strip()) > 50]

    if sections:
        chosen = random.choice(sections)
        return (
            "FRESH LAYOUT INSPIRATION (from design scout — use this as structural guidance):\n\n"
            f"{chosen}\n\n"
            "Adapt this layout concept to the client's content. Keep the structural approach but use the client's colors and fonts."
        )

    return None


async def _download_page_image(url: str) -> Optional[bytes]:
    """Download the main design image from a URL.

    Tries: og:image meta tag → first large image on page → screenshot-like fallback.
    Returns raw image bytes or None.
    """
    import re

    if not url:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,image/*",
    }

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
            # Check if the URL is already a direct image link
            if any(url.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                resp = await client.get(url)
                if resp.status_code == 200 and len(resp.content) > 5000:
                    logger.info(f"Downloaded direct image: {url[:80]} ({len(resp.content) // 1024}KB)")
                    return resp.content
                return None

            # Fetch the page HTML
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch page {url}: HTTP {resp.status_code}")
                return None

            html = resp.text

            # Try og:image first (most reliable for design sites)
            og_match = re.search(
                r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
                html, re.IGNORECASE
            )
            if not og_match:
                # Try reverse order (content before property)
                og_match = re.search(
                    r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']',
                    html, re.IGNORECASE
                )

            # Also try twitter:image
            if not og_match:
                og_match = re.search(
                    r'<meta[^>]*(?:name|property)=["\']twitter:image["\'][^>]*content=["\']([^"\']+)["\']',
                    html, re.IGNORECASE
                )

            image_url = None
            if og_match:
                image_url = og_match.group(1)
                # Handle relative URLs
                if image_url.startswith("//"):
                    image_url = "https:" + image_url
                elif image_url.startswith("/"):
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    image_url = f"{parsed.scheme}://{parsed.netloc}{image_url}"

            # Fallback: find large images in the HTML (Behance, Dribbble, Pinterest)
            if not image_url:
                # Look for large image URLs in common design portfolio patterns
                img_patterns = [
                    r'<img[^>]+src=["\']([^"\']+(?:1200|1400|1600|2000|large|original|full)[^"\']*)["\']',
                    r'srcset=["\'][^"\']*\s+(\S+)\s+\d{3,4}w["\']',  # largest srcset
                    r'"imageUrl"\s*:\s*"([^"]+)"',  # JSON embedded images
                    r'"url"\s*:\s*"(https?://[^"]+\.(?:jpg|jpeg|png|webp))"',
                ]
                for pattern in img_patterns:
                    match = re.search(pattern, html, re.IGNORECASE)
                    if match:
                        image_url = match.group(1)
                        if image_url.startswith("//"):
                            image_url = "https:" + image_url
                        break

            if not image_url:
                logger.warning(f"No image found on page: {url[:80]}")
                return None

            # Download the image
            img_resp = await client.get(image_url)
            if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                logger.info(f"Downloaded design image: {image_url[:80]} ({len(img_resp.content) // 1024}KB)")
                return img_resp.content
            else:
                logger.warning(f"Image download failed or too small: {image_url[:80]}")
                return None

    except Exception as e:
        logger.error(f"Failed to download image from {url[:80]}: {e}")
        return None


async def _save_scout_image_to_drive(image_bytes: bytes, item_name: str, item_url: str = ""):
    """Save a scout-downloaded image to Google Drive for permanent visual library."""
    import re
    import time

    # Get or create the scout folder
    scout_folder_id = config.DRIVE_SCOUT_FOLDER_ID
    if not scout_folder_id:
        # Create a "Scout Inspiration" subfolder under the templates folder
        parent = config.DRIVE_TEMPLATES_FOLDER_ID
        if not parent:
            logger.warning("No Drive folder configured for scout images")
            return
        from brain.drive_client import ensure_subfolder
        scout_folder_id = await ensure_subfolder(parent, "Scout Inspiration")
        logger.info(f"Created Scout Inspiration folder in Drive: {scout_folder_id}")

    # Clean filename from item name
    clean_name = re.sub(r'[^\w\s-]', '', item_name).strip().replace(' ', '_')[:40]
    timestamp = int(time.time())
    filename = f"scout_{timestamp}_{clean_name}.jpg"

    from brain.drive_client import upload_image
    file_id = await upload_image(scout_folder_id, filename, image_bytes)
    logger.info(f"Saved scout image to Drive: {filename} (id={file_id})")


async def extract_single_reference(pending: dict, item_index: int) -> Optional[dict]:
    """Extract a full forced_reference for a scout item — downloads the image + runs Vision analysis.

    Used when user says 'make a post like number 3'.
    Returns a dict compatible with forced_reference (same format as analyze_inspiration),
    with _image_b64 attached so Opus can SEE the layout.

    Falls back to text-only blueprint if image download fails.
    """
    items = pending.get("items", [])
    raw_results = pending.get("raw_results", [])

    # Find the item
    target = None
    for item in items:
        if item["index"] == item_index:
            target = item
            break

    if not target:
        return None

    item_url = target.get("url", "")
    item_name = target.get("name", "design")
    item_desc = target.get("description", "")
    # Serper gives us the direct image URL — use it first, fall back to page scraping
    direct_image_url = target.get("image_url", "")

    logger.info(f"Extracting reference for scout item #{item_index}: {item_name}")

    # Step 1: Download the design image — Serper already gave us the direct URL
    image_bytes = None
    if direct_image_url:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http:
                resp = await http.get(direct_image_url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                })
                if resp.status_code == 200 and len(resp.content) > 5000:
                    image_bytes = resp.content
                    logger.info(f"Downloaded direct image ({len(image_bytes)//1024}KB): {direct_image_url[:80]}")
        except Exception as e:
            logger.warning(f"Direct image download failed: {e}")

    # Fallback: try scraping og:image from the page URL
    if not image_bytes and item_url:
        image_bytes = await _download_page_image(item_url)

    if image_bytes:
        # Step 2: Save temporarily, run Vision analysis, and upload to Drive
        temp_path = None
        try:
            # Save to temp file for Vision
            suffix = ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(image_bytes)
                temp_path = f.name

            # Run the same Vision analysis used for inspiration photos
            from taste.vision import analyze_inspiration
            analysis = await analyze_inspiration(
                temp_path,
                context=f"Design reference from scout: {item_name} — {item_desc}",
            )

            # Attach the image bytes as base64 (so Opus can SEE the layout during template generation)
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            analysis["_image_b64"] = img_b64
            analysis["_source"] = f"design_scout: {item_name}"
            analysis["_url"] = item_url

            # Save to Google Drive for permanent visual library
            try:
                await _save_scout_image_to_drive(image_bytes, item_name, item_url)
            except Exception as e:
                logger.warning(f"Drive upload failed (non-critical): {e}")

            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

            logger.info(f"Full Vision analysis complete for scout item #{item_index}")
            return analysis

        except Exception as e:
            logger.warning(f"Vision analysis failed for scout image, falling back to text: {e}")
            # Clean up
            if temp_path:
                try:
                    Path(temp_path).unlink()
                except Exception:
                    pass

    # Fallback: Build a text-only reference dict (weaker but still better than nothing)
    logger.info(f"No image available for scout item #{item_index} — building text-only reference")

    combined = ""
    for label, result in raw_results:
        if "content" in result:
            combined += f"\n\n--- {label} ---\n{result['content']}"

    _ai = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = _ai.messages.create(
        model=config.SONNET_MODEL,
        max_tokens=1500,
        system=(
            "You extract a design analysis from research data. "
            "Return valid JSON matching this structure exactly:\n"
            "{\n"
            '  "composition": {"template_match": str, "text_position": str, "negative_space_pct": int, "visual_hierarchy": str},\n'
            '  "typography": [{"role": str, "font_category": str, "weight": str, "case": str, "tracking": str}],\n'
            '  "colors": {"temperature": str, "palette_mood": str},\n'
            '  "layers": [{"order": int, "type": str, "description": str, "position": str, "size": str}],\n'
            '  "feeling": {"mood": str, "communicates": str},\n'
            '  "reusable_rules": [str]\n'
            "}\n"
            "Be specific and precise. Return ONLY valid JSON."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Build a design analysis for: {item_name} — {item_desc}\n\n"
                f"Research context:\n{combined[:3000]}"
            ),
        }],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

    try:
        analysis = json.loads(raw_text)
        analysis["_source"] = f"design_scout (text-only): {item_name}"
        analysis["_url"] = item_url
        # No _image_b64 — Opus won't see the image but will get structured reference
        logger.info(f"Text-only reference built for scout item #{item_index}")
        return analysis
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse text reference: {e}")
        return None


# Keep the old function name as an alias for backwards compatibility
async def extract_single_blueprint(pending: dict, item_index: int) -> Optional[str]:
    """DEPRECATED: Use extract_single_reference() instead.
    Kept for backwards compatibility — returns text blueprint."""
    ref = await extract_single_reference(pending, item_index)
    if not ref:
        return None
    # Convert to text for old callers
    from pipeline.steps.dynamic_template import _format_reference
    text = _format_reference(ref, source=ref.get("_source", "scout"))
    return (
        "FORCED LAYOUT BLUEPRINT (user chose this specific design from scout results — FOLLOW IT CLOSELY):\n\n"
        f"{text}\n\n"
        "Build the HTML template following this exact structural approach. "
        "Adapt colors and fonts to the client's brand, but keep the layout structure identical."
    )
