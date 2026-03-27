"""Step 3: Creative Concept — Opus decides what the hero image should be."""

import json
import logging

import anthropic

import config
from pipeline.types import (
    PipelineInput, ResearchResult, BrainContext, CreativeConcept
)

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

CONCEPT_SYSTEM = """You are a senior creative director at an advertising agency.

Your ONE job: decide what the hero image should be for a social media post.

You think in visual metaphors — one unexpected object that implies the brand message
without explaining it. The viewer connects the dots.

Rules:
- ONE object only — never two competing focal points
- The object must be unexpected, not obvious (no lightbulbs for "ideas", no targets for "goals")
- The object carries the concept through implication, not illustration
- Think about what emotion the object evokes before anything else
- Consider the accumulated taste preferences — they define the visual language

You do NOT:
- Write any copy or headlines
- Pick fonts or colors
- Generate image prompts
- Make layout decisions

Return ONLY valid JSON:
{
  "object": "description of the single hero object",
  "why": "one sentence — why this object works for this brief",
  "emotional_direction": "the feeling this should evoke",
  "composition_note": "how the object sits in the frame",
  "background": "hex color or description",
  "lighting": "lighting description",
  "what_to_avoid": "what NOT to do with this image"
}"""


async def creative_concept(
    input: PipelineInput,
    research: ResearchResult,
    brain_ctx: BrainContext,
) -> CreativeConcept:
    """
    Generate a visual concept for the post.

    Input: research trends + brain context + taste data
    Output: CreativeConcept (object, why, emotional direction, etc.)
    """
    # Build the user message with all context
    taste = brain_ctx.taste_context
    past = brain_ctx.past_concepts

    user_msg = f"""Generate a visual concept for this post.

Client: {input.client}
Industry: {brain_ctx.brand.get('industry', 'general')}
Platform: {input.platform}
Brief: {input.brief}

Current trends:
{research.trends[:800]}

Brand tone: {brain_ctx.brand.get('tone', 'professional')}
Target audience: {brain_ctx.brand.get('target_audience', 'professionals')}

TASTE PREFERENCES (from analyzed inspiration):
  Preferred moods: {', '.join(taste.get('preferred_moods', ['quiet confidence']))}
  Preferred compositions: {', '.join(taste.get('preferred_compositions', ['object-hero']))}
  Confirmed rules: {', '.join(taste.get('confirmed_rules', ['one object, one idea'])[:5])}
  Avoid: {', '.join(taste.get('avoid', []))}

PAST CONCEPTS (DO NOT REPEAT):
{', '.join(past[:15]) if past else 'None yet — this is the first post.'}

Return only the JSON object."""

    try:
        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=1024,
            system=CONCEPT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)

        concept = CreativeConcept(
            object=data.get("object", ""),
            why=data.get("why", ""),
            emotional_direction=data.get("emotional_direction", ""),
            composition_note=data.get("composition_note", ""),
            background=data.get("background", "#F5F4F0"),
            lighting=data.get("lighting", "soft studio daylight"),
            what_to_avoid=data.get("what_to_avoid", ""),
        )

        logger.info(f"Concept: {concept.object} — {concept.why}")
        return concept

    except Exception as e:
        logger.error(f"Concept generation failed: {e}")
        raise
