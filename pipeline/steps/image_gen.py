"""Step 5: Image Generation — Opus writes prompt → Flux/Ideogram generates image."""

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


async def generate_image(
    concept: CreativeConcept,
    brain_ctx: BrainContext,
) -> ImageResult:
    """
    Write image prompt with Opus, then generate with Flux or Ideogram.

    Input: concept + taste context
    Output: ImageResult (path to downloaded image)
    """
    # Step 5a: Opus writes the prompt
    taste = brain_ctx.taste_context

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

        logger.info(f"Image spec: model={model}, prompt={len(prompt)} chars")

    except Exception as e:
        logger.error(f"Prompt writing failed: {e}")
        raise

    # Step 5b: Generate the image
    # Prefer the model Opus chose, but fall back if API key is missing
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
        # Submit generation
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

        # fal.ai returns a request_id for async — poll for result
        request_id = data.get("request_id")
        if request_id:
            result = await _poll_fal_result(client, request_id)
        else:
            result = data

        # Get image URL
        images = result.get("images", [])
        if not images:
            raise ValueError("Flux returned no images")

        image_url = images[0].get("url", "")

        # Download image
        img_response = await client.get(image_url)
        img_response.raise_for_status()

        # Save to output
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
            # Fetch result
            result_response = await client.get(
                f"https://queue.fal.run/fal-ai/flux-pro/v1.1/requests/{request_id}",
                headers={"Authorization": f"Key {config.FAL_KEY}"},
            )
            return result_response.json()
        elif status in ("FAILED", "CANCELLED"):
            raise ValueError(f"Flux generation {status}: {data}")

        await asyncio.sleep(2)

    raise TimeoutError("Flux generation timed out")


async def _generate_ideogram(prompt: str, spec: dict) -> ImageResult:
    """Generate image with Ideogram 3 API."""
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
                    "aspect_ratio": spec.get("aspect_ratio", "ASPECT_1_1"),
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

        # Download
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
