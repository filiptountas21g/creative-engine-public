"""Step 2: Brain Read — fetch client profile, past concepts, and taste data."""

import json
import logging

from brain.client import Brain
from taste.memory import get_taste_context
from pipeline.types import PipelineInput, BrainContext

logger = logging.getLogger(__name__)


async def brain_read(input: PipelineInput, brain: Brain) -> BrainContext:
    """
    Fetch all context needed for creative generation.

    Input: PipelineInput (client name)
    Output: BrainContext (brand, past_concepts, taste_context, client_profile)
    """
    ctx = BrainContext()

    # 1. Client profile
    try:
        clients = brain.get_clients(active_only=True)
        for c in clients:
            if c["name"].lower() == input.client.lower():
                ctx.client_profile = c
                ctx.brand = {
                    "tone": c.get("tone", "professional"),
                    "industry": c.get("industry", ""),
                    "target_audience": c.get("target_audience", ""),
                }
                break

        if not ctx.client_profile:
            logger.warning(f"Client '{input.client}' not found in Brain — using defaults")
            ctx.brand = {"tone": "professional", "industry": "general"}
    except Exception as e:
        logger.error(f"Failed to fetch client profile: {e}")

    # 2. Past concepts (for anti-repetition)
    try:
        past = brain.query(
            client=input.client,
            topic="generated_concept",
            limit=15,
        )
        ctx.past_concepts = []
        for entry in past:
            try:
                data = json.loads(entry["content"])
                ctx.past_concepts.append(data.get("object", entry.get("summary", "")))
            except (json.JSONDecodeError, KeyError):
                if entry.get("summary"):
                    ctx.past_concepts.append(entry["summary"])
    except Exception as e:
        logger.error(f"Failed to fetch past concepts: {e}")

    # 3. Taste context (accumulated design preferences)
    try:
        ctx.taste_context = await get_taste_context(brain, client=input.client)
    except Exception as e:
        logger.error(f"Failed to fetch taste context: {e}")
        ctx.taste_context = {}

    logger.info(
        f"Brain read: client={'found' if ctx.client_profile else 'missing'}, "
        f"past_concepts={len(ctx.past_concepts)}, "
        f"taste_refs={ctx.taste_context.get('total_references', 0)}"
    )

    return ctx
