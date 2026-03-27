"""Step 8: Brain Write — store the generated concept so it's never repeated."""

import json
import logging

from brain.client import Brain
from pipeline.types import (
    PipelineInput, CreativeConcept, CopyOptions,
    CreativeDecisions, ImageResult,
)

logger = logging.getLogger(__name__)


async def brain_write(
    input: PipelineInput,
    concept: CreativeConcept,
    copy: CopyOptions,
    decisions: CreativeDecisions,
    image: ImageResult,
    brain: Brain,
) -> None:
    """
    Store the completed concept in Big Brain for anti-repetition.

    Every generated post is stored so Opus never repeats the same concept.
    """
    entry = {
        "object": concept.object,
        "why": concept.why,
        "emotional_direction": concept.emotional_direction,
        "brief": input.brief,
        "headline": decisions.headline,
        "font": decisions.font_headline,
        "template": decisions.template,
        "image_model": image.model_used,
        "color_bg": decisions.color_bg,
        "platform": input.platform,
    }

    try:
        brain.store(
            topic="generated_concept",
            source="creative_engine",
            content=json.dumps(entry, ensure_ascii=False),
            client=input.client,
            summary=f"{concept.object} — {concept.why}",
            tags=[
                "generated",
                decisions.template,
                image.model_used,
            ],
        )
        logger.info(f"Stored concept in Brain: {concept.object}")
    except Exception as e:
        logger.error(f"Failed to store concept in Brain: {e}")
        # Non-fatal — the post was still generated
