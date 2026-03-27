"""Step 4: Copy — Sonnet writes 3 headlines + subtext + CTA."""

import json
import logging

import anthropic

import config
from pipeline.types import PipelineInput, CreativeConcept, BrainContext, CopyOptions

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

COPY_SYSTEM = """You are a copywriter for a Greek advertising agency.

You write sharp, witty, minimal copy for LinkedIn and social media posts.
Your copy completes the visual concept — the image implies the idea,
your headline confirms it with wit.

Rules:
- Headlines: under 8 words. Punchy. No fluff.
- Write in the language appropriate for the client and platform.
  Greek clients on LinkedIn: write in Greek.
- The headline should make the viewer think, not tell them what to think.
- Subtext: one supporting sentence, max 15 words.
- CTA: max 5 words.

You do NOT:
- Make visual decisions
- Pick fonts or colors
- Describe layout or composition

Return ONLY valid JSON:
{
  "headlines": ["option 1", "option 2", "option 3"],
  "subtext": "one supporting sentence",
  "cta": "call to action"
}"""


async def write_copy(
    input: PipelineInput,
    concept: CreativeConcept,
    brain_ctx: BrainContext,
) -> CopyOptions:
    """
    Write copy for the post — 3 headline options + subtext + CTA.

    Uses Sonnet (fast, cheap, sharp for copy).
    Does NOT see visual direction fields — only concept + tone.
    """
    user_msg = f"""Write copy for this post.

Client: {input.client}
Brief: {input.brief}
Visual concept: {concept.object} — {concept.why}
Emotional direction: {concept.emotional_direction}
Brand tone: {brain_ctx.brand.get('tone', 'professional')}
Platform: {input.platform}
Target audience: {brain_ctx.brand.get('target_audience', 'professionals')}

Return the JSON with 3 headline options, subtext, and CTA."""

    try:
        response = _client.messages.create(
            model=config.SONNET_MODEL,
            max_tokens=1024,
            system=COPY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)

        copy = CopyOptions(
            headlines=data.get("headlines", [])[:3],
            subtext=data.get("subtext", ""),
            cta=data.get("cta", ""),
        )

        logger.info(f"Copy: {len(copy.headlines)} headlines, subtext={len(copy.subtext)} chars")
        return copy

    except Exception as e:
        logger.error(f"Copy generation failed: {e}")
        raise
