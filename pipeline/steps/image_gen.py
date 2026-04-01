"""Step 5: Image Sourcing — stock photos first, AI generation as fallback.

Flow:
1. Opus writes a search query for stock photos
2. Search Unsplash + Pexels
3. Claude Vision judges the top results
4. If a stock photo is good enough → use it
5. If not → Opus writes an AI prompt → Ideogram/Flux generates
"""

import base64
import json
import logging
import tempfile
from pathlib import Path

import anthropic
import httpx

import config
from pipeline.types import CreativeConcept, BrainContext, ImageResult

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# ── Search query writer ──────────────────────────────────

SEARCH_WRITER_SYSTEM = """You write search queries for stock photo sites.
Given a visual concept, return the best search terms to find a matching photo.

Return ONLY valid JSON:
{
  "queries": ["primary search query", "alternative query"],
  "style": "editorial/minimal/moody/bright/corporate",
  "must_have": "key element that must be in the photo"
}

Keep queries short and specific (2-4 words). Focus on the main subject.
Example: concept="vintage compass on leather" → queries=["vintage brass compass", "compass leather case"]"""


async def generate_image(
    concept: CreativeConcept,
    brain_ctx: BrainContext,
    image_source: str = "auto",
) -> ImageResult:
    """
    Source the best image for this concept.

    image_source:
      "auto" — tries stock first, falls back to AI (default)
      "stock" — stock photos only, lower quality threshold, no AI fallback
      "ai" — AI generation only, skip stock search
    """
    taste = brain_ctx.taste_context

    logger.info(f"Image generation starting: image_source={image_source}")

    if image_source == "ai":
        logger.info("AI-only mode — skipping stock search")
        return await _generate_ai_image(concept, taste)

    # Try stock photos if APIs are available
    has_unsplash = bool(config.UNSPLASH_ACCESS_KEY)
    has_pexels = bool(config.PEXELS_API_KEY)
    logger.info(f"Stock APIs available: Unsplash={has_unsplash}, Pexels={has_pexels}")

    if has_unsplash or has_pexels:
        try:
            # Lower the bar for stock-only mode
            is_stock_mode = image_source == "stock"
            logger.info(f"Searching stock photos (lenient={is_stock_mode})...")
            stock_result = await _try_stock_photos(concept, taste, lenient=is_stock_mode)
            if stock_result:
                logger.info(f"✓ Stock photo accepted: {stock_result.model_used}")
                return stock_result
            if is_stock_mode:
                logger.warning("Stock-only mode — first search failed, trying broader search...")
                stock_result = await _try_stock_photos_broad(concept, taste)
                if stock_result:
                    logger.info(f"✓ Broad stock search found: {stock_result.model_used}")
                    return stock_result
                logger.warning("Stock-only mode — ALL stock attempts failed, forced to use AI")
            else:
                logger.info("Stock photos not good enough in auto mode — falling back to AI")
        except Exception as e:
            logger.warning(f"Stock photo search failed with exception: {e}")

    elif image_source == "stock":
        logger.error("User requested stock photos but NO stock API keys are configured!")

    # Fallback: AI generation
    logger.info("Generating AI image...")
    return await _generate_ai_image(concept, taste)


# ── Stock photo pipeline ──────────────────────────────────

async def _try_stock_photos(concept: CreativeConcept, taste: dict, lenient: bool = False) -> ImageResult | None:
    """Search stock photos, judge them with Claude Vision, return best or None.
    If lenient=True, accept medium/low confidence and be less strict about exact concept match."""

    # Step 1: Get search queries from Opus
    search_spec = await _write_search_queries(concept)
    queries = search_spec.get("queries", [concept.object])
    must_have = search_spec.get("must_have", "")

    # Step 2: Search both sources
    candidates = []
    async with httpx.AsyncClient(timeout=30) as http:
        for query in queries[:2]:  # max 2 queries
            logger.info(f"Searching stock photos for: '{query}'")
            if config.UNSPLASH_ACCESS_KEY:
                unsplash_results = await _search_unsplash(http, query)
                logger.info(f"  Unsplash returned {len(unsplash_results)} results for '{query}'")
                candidates.extend(unsplash_results)
            if config.PEXELS_API_KEY:
                pexels_results = await _search_pexels(http, query)
                logger.info(f"  Pexels returned {len(pexels_results)} results for '{query}'")
                candidates.extend(pexels_results)

    logger.info(f"Total stock candidates found: {len(candidates)}")
    if not candidates:
        logger.warning("No stock photo results from any source")
        return None

    # Step 3: Download top candidates for Vision judging
    # Take top 6 for better selection
    top_candidates = candidates[:6]

    async with httpx.AsyncClient(timeout=30) as http:
        downloaded = []
        for cand in top_candidates:
            try:
                resp = await http.get(cand["thumb_url"])
                resp.raise_for_status()
                downloaded.append({
                    **cand,
                    "image_bytes": resp.content,
                })
            except Exception:
                continue

    if not downloaded:
        return None

    # Step 4: Claude Vision judges
    best = await _judge_stock_photos(downloaded, concept, must_have, taste, lenient=lenient)

    if not best:
        return None

    # Step 5: Download the full-resolution winner
    async with httpx.AsyncClient(timeout=60) as http:
        resp = await http.get(best["full_url"])
        resp.raise_for_status()

        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / "hero_temp.png"
        img_path.write_bytes(resp.content)

        logger.info(f"Stock photo saved: {img_path} (source: {best['source']})")

        return ImageResult(
            image_path=str(img_path),
            image_url=best["full_url"],
            model_used=f"stock-{best['source']}",
            prompt_used=f"Stock search: {best.get('query', '')}",
        )


async def _try_stock_photos_broad(concept: CreativeConcept, taste: dict) -> ImageResult | None:
    """Broader stock search as a second attempt — uses simpler, more generic queries."""
    # Extract the core subject/mood as simpler search terms
    simple_queries = [
        concept.object.split(",")[0].strip()[:30],  # First part of concept only
        concept.emotional_direction or "professional",
    ]

    candidates = []
    async with httpx.AsyncClient(timeout=30) as http:
        for query in simple_queries:
            if config.UNSPLASH_ACCESS_KEY:
                results = await _search_unsplash(http, query)
                candidates.extend(results)

    if not candidates:
        return None

    top_candidates = candidates[:6]  # More candidates for broader search
    async with httpx.AsyncClient(timeout=30) as http:
        downloaded = []
        for cand in top_candidates:
            try:
                resp = await http.get(cand["thumb_url"])
                resp.raise_for_status()
                downloaded.append({**cand, "image_bytes": resp.content})
            except Exception:
                continue

    if not downloaded:
        return None

    best = await _judge_stock_photos(downloaded, concept, "", taste, lenient=True)
    if not best:
        return None

    async with httpx.AsyncClient(timeout=60) as http:
        resp = await http.get(best["full_url"])
        resp.raise_for_status()
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / "hero_temp.png"
        img_path.write_bytes(resp.content)
        logger.info(f"Broad stock search found: {img_path} (source: {best['source']})")
        return ImageResult(
            image_path=str(img_path),
            image_url=best["full_url"],
            model_used=f"stock-{best['source']}",
            prompt_used=f"Broad stock search: {best.get('query', '')}",
        )


async def _write_search_queries(concept: CreativeConcept) -> dict:
    """Use Sonnet to write stock photo search queries."""
    try:
        response = _client.messages.create(
            model=config.SONNET_MODEL,
            max_tokens=200,
            system=SEARCH_WRITER_SYSTEM,
            messages=[{"role": "user", "content": (
                f"Concept: {concept.object}\n"
                f"Mood: {concept.emotional_direction}\n"
                f"Background: {concept.background}\n"
                f"Lighting: {concept.lighting}"
            )}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Search query writing failed: {e}")
        # Fallback: use concept object as query
        return {"queries": [concept.object], "must_have": concept.object}


async def _search_unsplash(http: httpx.AsyncClient, query: str) -> list[dict]:
    """Search Unsplash for photos."""
    try:
        resp = await http.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": query,
                "per_page": 8,
                "orientation": "squarish",
            },
            headers={"Authorization": f"Client-ID {config.UNSPLASH_ACCESS_KEY}"},
        )
        if resp.status_code != 200:
            logger.warning(f"Unsplash returned HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()

        results = []
        for photo in data.get("results", []):
            results.append({
                "source": "unsplash",
                "query": query,
                "id": photo["id"],
                "thumb_url": photo["urls"]["small"],  # 400px for judging
                "full_url": photo["urls"]["regular"],  # 1080px for final
                "description": photo.get("description") or photo.get("alt_description") or "",
                "photographer": photo["user"]["name"],
            })
        return results
    except Exception as e:
        logger.warning(f"Unsplash search exception: {e}")
        return []


async def _search_pexels(http: httpx.AsyncClient, query: str) -> list[dict]:
    """Search Pexels for photos."""
    try:
        resp = await http.get(
            "https://api.pexels.com/v1/search",
            params={
                "query": query,
                "per_page": 8,
                "size": "medium",
            },
            headers={"Authorization": config.PEXELS_API_KEY},
        )
        if resp.status_code != 200:
            logger.warning(f"Pexels returned HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()

        results = []
        for photo in data.get("photos", []):
            results.append({
                "source": "pexels",
                "query": query,
                "id": photo["id"],
                "thumb_url": photo["src"]["medium"],  # ~400px for judging
                "full_url": photo["src"]["large2x"],  # ~1200px for final
                "description": photo.get("alt") or "",
                "photographer": photo.get("photographer", ""),
            })
        return results
    except Exception as e:
        logger.warning(f"Pexels search exception: {e}")
        return []


async def _judge_stock_photos(
    candidates: list[dict],
    concept: CreativeConcept,
    must_have: str,
    taste: dict,
    lenient: bool = False,
) -> dict | None:
    """Use Claude Vision to judge stock photos. Returns the best one or None."""

    strictness = (
        "IMPORTANT: The user specifically asked for STOCK PHOTOS. You MUST pick one. "
        "Choose the BEST available photo. ANY decent professional photo is acceptable. "
        "It does NOT need to match the concept exactly — general mood/vibe is enough. "
        "Return -1 ONLY if every single photo is extremely low quality or completely wrong "
        "(e.g. all photos are of food when concept is about technology). "
        "When in doubt, PICK ONE. The user prefers a real photo over AI-generated."
    ) if lenient else (
        "Be SELECTIVE — only pick a photo if it genuinely matches the concept. "
        "If the results are generic or don't capture the specific idea, return -1 "
        "so we can generate with AI instead."
    )

    # Build vision message with all candidate images
    content = [
        {
            "type": "text",
            "text": (
                f"I'm looking for a stock photo to use in a social media post.\n\n"
                f"CONCEPT: {concept.object}\n"
                f"MOOD: {concept.emotional_direction}\n"
                f"MUST HAVE: {must_have}\n"
                f"BACKGROUND PREFERENCE: {concept.background}\n"
                f"LIGHTING PREFERENCE: {concept.lighting}\n"
                f"PREFERRED MOODS: {', '.join(taste.get('preferred_moods', []))}\n\n"
                f"Below are {len(candidates)} stock photo candidates. For each one, judge:\n"
                f"1. Does it match the concept? (the main subject must be clearly present)\n"
                f"2. Is the quality high enough? (sharp, well-lit, professional)\n"
                f"3. Does it feel right for the mood?\n"
                f"4. Can it work as a hero image in a designed post?\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"winner": 0-based index of best photo or -1 if NONE are good enough, '
                f'"reason": "why this one (or why none work)", '
                f'"confidence": "high/medium/low"}}\n\n'
                f"{strictness}"
            ),
        }
    ]

    for i, cand in enumerate(candidates):
        # Add image
        img_b64 = base64.b64encode(cand["image_bytes"]).decode("utf-8")
        content.append({
            "type": "text",
            "text": f"\nPhoto {i} ({cand['source']}): {cand.get('description', 'no description')}",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_b64,
            },
        })

    try:
        response = _client.messages.create(
            model=config.VISION_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": content}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)
        winner_idx = result.get("winner", -1)
        confidence = result.get("confidence", "low")
        reason = result.get("reason", "")

        logger.info(f"Stock photo judge: winner={winner_idx}, confidence={confidence}, reason={reason[:80]}")

        if winner_idx < 0 or winner_idx >= len(candidates):
            return None

        # In lenient mode, accept any confidence; in strict mode reject low
        if confidence == "low" and not lenient:
            logger.info("Low confidence — rejecting stock photos")
            return None

        return candidates[winner_idx]

    except Exception as e:
        logger.error(f"Stock photo judging failed: {e}")
        return None


# ── AI generation pipeline (fallback) ─────────────────────

PROMPT_WRITER_SYSTEM = """You are an expert at writing image generation prompts.

You receive a visual concept and write the optimal prompt for AI image generation.

Rules for Flux 2 (photorealistic):
- Put the main subject in the FIRST sentence — Flux weights early tokens more
- Single subject dramatically increases quality
- Structure: [subject] + [positioning] + [background] + [lighting] + [style]
- NEVER include text in the prompt — text will be added by the template system
- Keep prompts under 200 words

Rules for Ideogram 3 (illustrated/graphic):
- Best for typography and text-heavy visuals
- Use when the concept requires text baked into the image
- Specify style clearly: editorial illustration, vector, etc.

Choose Flux 2 for:
- Photorealistic objects, product shots, editorial photography
- Single objects on clean backgrounds (default choice)

Choose Ideogram 3 for:
- Illustrated style, graphic design, when text must appear in the image

Return ONLY valid JSON:
{
  "model": "flux2" or "ideogram3",
  "prompt": "the generation prompt",
  "aspect_ratio": "1:1",
  "negative_prompt": "what to avoid in the image"
}"""


async def _generate_ai_image(concept: CreativeConcept, taste: dict) -> ImageResult:
    """Generate image with AI (Flux or Ideogram) — the original pipeline."""

    user_msg = f"""Write an image generation prompt for this concept.

Object: {concept.object}
Why: {concept.why}
Emotional direction: {concept.emotional_direction}
Composition: {concept.composition_note}
Background: {concept.background}
Lighting: {concept.lighting}
Avoid: {concept.what_to_avoid}

Taste preferences:
  Preferred compositions: {', '.join(taste.get('preferred_compositions', ['object-hero']))}
  Preferred moods: {', '.join(taste.get('preferred_moods', ['quiet confidence']))}
  Avoid: {', '.join(taste.get('avoid', []))}

IMPORTANT: Do NOT include any text in the prompt. The template system handles all text."""

    try:
        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=1024,
            system=PROMPT_WRITER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        spec = json.loads(raw)
        model = spec.get("model", "flux2")
        prompt = spec.get("prompt", "")
        negative = spec.get("negative_prompt", "")

        logger.info(f"AI image spec: model={model}, prompt={len(prompt)} chars")

    except Exception as e:
        logger.error(f"Prompt writing failed: {e}")
        raise

    # Generate the image
    if model == "flux2" and config.FAL_KEY:
        return await _generate_flux(prompt, negative, spec)
    elif config.IDEOGRAM_API_KEY:
        return await _generate_ideogram(prompt, spec)
    elif config.FAL_KEY:
        return await _generate_flux(prompt, negative, spec)
    else:
        raise ValueError("No image generation API key set (need FAL_KEY or IDEOGRAM_API_KEY)")


async def _generate_flux(prompt: str, negative: str, spec: dict) -> ImageResult:
    """Generate image with Flux 2 via fal.ai."""
    if not config.FAL_KEY:
        raise ValueError("FAL_KEY not set — cannot generate Flux images")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://queue.fal.run/fal-ai/flux-pro/v1.1",
            headers={
                "Authorization": f"Key {config.FAL_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt,
                "image_size": {"width": 1080, "height": 1080},
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "num_images": 1,
                "enable_safety_checker": True,
            },
        )
        response.raise_for_status()
        data = response.json()

        request_id = data.get("request_id")
        if request_id:
            result = await _poll_fal_result(client, request_id)
        else:
            result = data

        images = result.get("images", [])
        if not images:
            raise ValueError("Flux returned no images")

        image_url = images[0].get("url", "")

        img_response = await client.get(image_url)
        img_response.raise_for_status()

        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / "hero_temp.png"
        img_path.write_bytes(img_response.content)

        logger.info(f"Flux image saved: {img_path}")

        return ImageResult(
            image_path=str(img_path),
            image_url=image_url,
            model_used="flux2",
            prompt_used=prompt,
        )


async def _poll_fal_result(client: httpx.AsyncClient, request_id: str, max_polls: int = 60) -> dict:
    """Poll fal.ai for async result."""
    import asyncio

    for _ in range(max_polls):
        response = await client.get(
            f"https://queue.fal.run/fal-ai/flux-pro/v1.1/requests/{request_id}/status",
            headers={"Authorization": f"Key {config.FAL_KEY}"},
        )
        data = response.json()
        status = data.get("status")

        if status == "COMPLETED":
            result_response = await client.get(
                f"https://queue.fal.run/fal-ai/flux-pro/v1.1/requests/{request_id}",
                headers={"Authorization": f"Key {config.FAL_KEY}"},
            )
            return result_response.json()
        elif status in ("FAILED", "CANCELLED"):
            raise ValueError(f"Flux generation {status}: {data}")

        await asyncio.sleep(2)

    raise TimeoutError("Flux generation timed out")


async def generate_extra_images(
    concept: CreativeConcept,
    brain_ctx: BrainContext,
    count: int,
    image_source: str = "auto",
) -> list[ImageResult]:
    """Generate multiple additional images for multi-image templates.

    For stock: does one batch search and picks the top N different photos.
    For AI: generates N separate images.
    Each image gets a unique filename (extra_1.png, extra_2.png, etc.).
    """
    taste = brain_ctx.taste_context

    # Try batch stock search first (unless AI-only)
    if image_source != "ai":
        has_stock = bool(config.UNSPLASH_ACCESS_KEY) or bool(config.PEXELS_API_KEY)
        if has_stock:
            try:
                logger.info(f"Batch stock search for {count} extra images...")
                stock_results = await _batch_stock_photos(concept, taste, count, lenient=(image_source == "stock"))
                if len(stock_results) >= count:
                    logger.info(f"Got {len(stock_results)} stock photos for extra images")
                    return stock_results[:count]
                elif stock_results:
                    logger.info(f"Got {len(stock_results)}/{count} stock photos, generating rest with AI")
                    # Fill remaining with AI
                    for i in range(len(stock_results), count):
                        try:
                            img = await _generate_ai_image(concept, taste)
                            unique_path = Path(img.image_path).parent / f"extra_{i + 1}.png"
                            import shutil
                            shutil.copy2(img.image_path, str(unique_path))
                            img.image_path = str(unique_path)
                            stock_results.append(img)
                        except Exception as e:
                            logger.error(f"AI fallback for extra image {i + 1} failed: {e}")
                    return stock_results
            except Exception as e:
                logger.warning(f"Batch stock search failed: {e}")

    # AI generation: generate each one separately
    results = []
    for i in range(count):
        try:
            logger.info(f"Generating extra AI image {i + 1}/{count}...")
            img = await _generate_ai_image(concept, taste)

            # Rename to unique path
            original = Path(img.image_path)
            unique_path = original.parent / f"extra_{i + 1}{original.suffix}"
            import shutil
            shutil.copy2(str(original), str(unique_path))
            img.image_path = str(unique_path)

            results.append(img)
            logger.info(f"Extra image {i + 1} ready: {img.image_path}")
        except Exception as e:
            logger.error(f"Extra image {i + 1} failed: {e}")

    return results


async def _batch_stock_photos(
    concept: CreativeConcept,
    taste: dict,
    count: int,
    lenient: bool = False,
) -> list[ImageResult]:
    """Search stock photos and return the top N different ones."""

    # Get search queries
    search_spec = await _write_search_queries(concept)
    queries = search_spec.get("queries", [concept.object])

    # Search with more queries to get a bigger pool
    candidates = []
    async with httpx.AsyncClient(timeout=30) as http:
        for query in queries[:3]:  # up to 3 queries for bigger pool
            logger.info(f"Batch stock search: '{query}'")
            if config.UNSPLASH_ACCESS_KEY:
                results = await _search_unsplash(http, query)
                candidates.extend(results)
            if config.PEXELS_API_KEY:
                results = await _search_pexels(http, query)
                candidates.extend(results)

    logger.info(f"Batch stock: {len(candidates)} total candidates for {count} slots")
    if not candidates:
        return []

    # Download thumbnails for judging — get more than we need
    top_candidates = candidates[:min(len(candidates), count * 4)]
    async with httpx.AsyncClient(timeout=30) as http:
        downloaded = []
        for cand in top_candidates:
            try:
                resp = await http.get(cand["thumb_url"])
                resp.raise_for_status()
                downloaded.append({**cand, "image_bytes": resp.content})
            except Exception:
                continue

    if not downloaded:
        return []

    # Ask Vision to rank the top N
    winners = await _judge_stock_photos_batch(downloaded, concept, taste, count, lenient=lenient)
    if not winners:
        return []

    # Download full-res for each winner
    results = []
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60) as http:
        for i, winner in enumerate(winners):
            try:
                resp = await http.get(winner["full_url"])
                resp.raise_for_status()
                img_path = output_dir / f"extra_{i + 1}.png"
                img_path.write_bytes(resp.content)
                results.append(ImageResult(
                    image_path=str(img_path),
                    image_url=winner["full_url"],
                    model_used=f"stock-{winner['source']}",
                    prompt_used=f"Stock batch: {winner.get('query', '')}",
                ))
                logger.info(f"Batch stock photo {i + 1} saved: {img_path}")
            except Exception as e:
                logger.error(f"Failed to download stock photo {i + 1}: {e}")

    return results


async def _judge_stock_photos_batch(
    candidates: list[dict],
    concept: CreativeConcept,
    taste: dict,
    count: int,
    lenient: bool = False,
) -> list[dict]:
    """Use Claude Vision to pick the top N different stock photos."""

    content = [
        {
            "type": "text",
            "text": (
                f"I need {count} DIFFERENT stock photos for a multi-image social media post.\n\n"
                f"CONCEPT: {concept.object}\n"
                f"MOOD: {concept.emotional_direction}\n\n"
                f"Below are {len(candidates)} candidates. Pick the BEST {count} that:\n"
                f"1. Are visually DIFFERENT from each other (variety matters)\n"
                f"2. Are professional quality\n"
                f"3. Work together as a cohesive set\n"
                f"4. Match the general mood/concept\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"winners": [0-based indices of the {count} best photos], '
                f'"reason": "why these work as a set"}}\n\n'
                f"{'Pick ANY decent professional photos. Be lenient.' if lenient else 'Be selective but you MUST pick ' + str(count) + '.'}"
            ),
        }
    ]

    for i, cand in enumerate(candidates):
        img_b64 = base64.b64encode(cand["image_bytes"]).decode("utf-8")
        content.append({
            "type": "text",
            "text": f"\nPhoto {i} ({cand['source']}): {cand.get('description', '')}",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_b64,
            },
        })

    try:
        response = _client.messages.create(
            model=config.VISION_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": content}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)
        winner_indices = result.get("winners", [])
        reason = result.get("reason", "")
        logger.info(f"Batch stock judge: winners={winner_indices}, reason={reason[:100]}")

        # Return the winning candidates in order
        winners = []
        for idx in winner_indices:
            if 0 <= idx < len(candidates):
                winners.append(candidates[idx])
        return winners

    except Exception as e:
        logger.error(f"Batch stock judging failed: {e}")
        return []


async def _generate_ideogram(prompt: str, spec: dict) -> ImageResult:
    """Generate image with Ideogram 3 API."""
    ASPECT_MAP = {
        "1:1": "ASPECT_1_1",
        "4:5": "ASPECT_4_5",
        "5:4": "ASPECT_5_4",
        "9:16": "ASPECT_9_16",
        "16:9": "ASPECT_16_9",
        "3:4": "ASPECT_3_4",
        "4:3": "ASPECT_4_3",
        "2:3": "ASPECT_2_3",
        "3:2": "ASPECT_3_2",
    }
    raw_ratio = spec.get("aspect_ratio", "1:1")
    aspect = ASPECT_MAP.get(raw_ratio, raw_ratio if raw_ratio.startswith("ASPECT_") else "ASPECT_1_1")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.ideogram.ai/generate",
            headers={
                "Api-Key": config.IDEOGRAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "image_request": {
                    "prompt": prompt,
                    "aspect_ratio": aspect,
                    "model": "V_2",
                    "magic_prompt_option": "OFF",
                },
            },
        )
        response.raise_for_status()
        data = response.json()

        images = data.get("data", [])
        if not images:
            raise ValueError("Ideogram returned no images")

        image_url = images[0].get("url", "")

        img_response = await client.get(image_url)
        img_response.raise_for_status()

        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        img_path = output_dir / "hero_temp.png"
        img_path.write_bytes(img_response.content)

        logger.info(f"Ideogram image saved: {img_path}")

        return ImageResult(
            image_path=str(img_path),
            image_url=image_url,
            model_used="ideogram3",
            prompt_used=prompt,
        )
