"""Step 2: Brain Read — fetch client profile, past concepts, and liked templates."""

import json
import logging

from brain.client import Brain
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

    # 3. Style direction — pull directly from liked templates (actual designs user approved)
    try:
        liked = brain.query(topic="liked_template", limit=20)
        # Also pull client-specific liked templates
        if input.client and input.client != "ALL":
            client_liked = brain.query(topic="liked_template", client=input.client, limit=10)
        else:
            client_liked = []

        # Extract concrete style preferences from liked templates
        fonts = []
        colors_bg = []
        colors_text = []
        colors_accent = []
        templates = []
        liked_image_styles = []

        for entry in liked:
            try:
                data = json.loads(entry["content"])
                if data.get("font_headline"):
                    fonts.append(f"{data['font_headline']} {data.get('font_headline_weight', 700)}")
                if data.get("color_bg"):
                    colors_bg.append(data["color_bg"])
                if data.get("color_text"):
                    colors_text.append(data["color_text"])
                if data.get("color_accent"):
                    colors_accent.append(data["color_accent"])
                if data.get("template_style"):
                    templates.append(data["template_style"])
                if data.get("image_description"):
                    liked_image_styles.append(data["image_description"])
            except (json.JSONDecodeError, KeyError):
                continue

        # Pull blacklisted styles (user said "delete this template" on a design they hated)
        disliked = brain.query(topic="disliked_template", limit=20)
        avoid_combos = []
        for entry in disliked:
            try:
                data = json.loads(entry["content"])
                combo = f"{data.get('template_style', '?')} + {data.get('font_headline', '?')}"
                avoid_combos.append(combo)
            except (json.JSONDecodeError, KeyError):
                continue

        # Build taste_context in the same format the pipeline expects
        ctx.taste_context = {
            "preferred_fonts": list(dict.fromkeys(fonts)),  # dedupe, keep order
            "preferred_color_temperatures": [],
            "preferred_colors": list(dict.fromkeys(colors_bg + colors_accent)),
            "preferred_compositions": list(dict.fromkeys(templates)),
            "preferred_moods": [],
            "confirmed_rules": [],
            "avoid": avoid_combos,
            "corrections": [],
            "client_specific": {},
            "total_references": len(liked),
            "liked_image_styles": liked_image_styles,  # what imagery the user approved
            "liked_templates_raw": [
                json.loads(e["content"]) for e in (client_liked or liked)
                if e.get("content")
            ],
        }

        if client_liked:
            client_fonts = []
            client_accents = []
            for entry in client_liked:
                try:
                    data = json.loads(entry["content"])
                    if data.get("font_headline"):
                        client_fonts.append(data["font_headline"])
                    if data.get("color_accent"):
                        client_accents.append(data["color_accent"])
                except (json.JSONDecodeError, KeyError):
                    continue
            if client_fonts or client_accents:
                ctx.taste_context["client_specific"][input.client] = (
                    f"Liked fonts: {', '.join(set(client_fonts))}. "
                    f"Liked accents: {', '.join(set(client_accents))}."
                )

    except Exception as e:
        logger.error(f"Failed to fetch liked templates: {e}")
        ctx.taste_context = {}

    logger.info(
        f"Brain read: client={'found' if ctx.client_profile else 'missing'}, "
        f"past_concepts={len(ctx.past_concepts)}, "
        f"liked_templates={ctx.taste_context.get('total_references', 0)}"
    )

    return ctx
