"""Step 6b: Dynamic Template — generate a fresh HTML layout each time from inspiration."""

import json
import logging
import random

import anthropic

import config
from brain.client import Brain
from pipeline.types import CreativeDecisions

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

TEMPLATE_SYSTEM = """You are an expert HTML/CSS designer building a social media post template.

You will receive:
1. A specific inspiration reference (composition, colors, typography, layers) from the user's taste library
2. The creative decisions (headline, font, colors, template style)

Your job: create a UNIQUE HTML layout inspired by the reference but adapted for this specific post.

The template MUST:
1. Be exactly 1080x1080px
2. Use CSS custom properties (variables) for ALL dynamic values
3. Have NO JavaScript
4. Be self-contained (inline CSS only, except Google Fonts)
5. Have clean, semantic HTML

CSS variables to use (they will be overridden by the render engine):
  --font-headline, --font-headline-weight, --font-headline-size,
  --font-headline-tracking, --font-headline-line-height, --font-headline-case
  --font-subtext, --font-subtext-weight, --font-subtext-size
  --color-bg, --color-text, --color-accent, --color-subtext
  --headline-margin-x, --headline-margin-y, --headline-max-width
  --image-padding, --image-object-fit
  --client-color

Placeholders (MANDATORY — the render engine replaces these):
  {{FONT_URL}} — Google Fonts link href
  {{HEADLINE}} — main headline text
  {{SUBTEXT}} — supporting text
  {{IMAGE_PATH}} — hero image (MUST be an <img> tag, NOT CSS background-image)
  {{CTA}} — call to action text
  {{CLIENT_NAME}} — client name for footer label

CRITICAL RULES:
- Every template MUST include <img src="{{IMAGE_PATH}}"> for the hero image
- Do NOT use background-image CSS for the hero
- Make each layout genuinely different — vary grid structures, text placement, image sizing, decorative elements
- Use the inspiration as a starting point, not a copy
- Be creative with layout: overlapping elements, asymmetric grids, text over image, sidebar layouts, diagonal cuts, etc.

Return ONLY the complete HTML file. No explanation, no markdown fences."""


async def generate_dynamic_template(
    decisions: CreativeDecisions,
    brain: Brain,
    previous_templates: list[str] | None = None,
) -> str:
    """
    Generate a fresh HTML template based on a random inspiration reference.

    Returns the HTML string ready for injection.
    """
    # Get all inspiration references
    refs = brain.query(topic="taste_reference", limit=50)

    # Get liked templates (higher priority)
    liked = brain.query(topic="liked_template", limit=20)

    # Pick a reference to inspire the layout
    reference_text = _pick_reference(refs, liked, decisions.template, previous_templates)

    # Build the prompt
    avoid_text = ""
    if previous_templates:
        avoid_text = f"\n\nAVOID these layouts (already used recently):\n"
        for i, desc in enumerate(previous_templates[-3:], 1):
            avoid_text += f"  {i}. {desc}\n"
        avoid_text += "Make something VISUALLY DIFFERENT from all of the above."

    user_prompt = f"""Create a unique HTML template for this post.

INSPIRATION REFERENCE (use as creative starting point):
{reference_text}

CREATIVE DECISIONS FOR THIS POST:
  Template style hint: {decisions.template}
  Headline: {decisions.headline}
  Font: {decisions.font_headline} ({decisions.font_headline_weight})
  Colors: bg={decisions.color_bg}, text={decisions.color_text}, accent={decisions.color_accent}

The template style hint is just a suggestion — you can interpret it freely.
For example, "split" doesn't have to be a strict 50/50 split — it could be 30/70, or angled, or overlapping.
{avoid_text}

Make this layout UNIQUE. Don't default to basic split or centered layouts every time.
Consider: asymmetric compositions, overlapping elements, creative image cropping,
bold geometric shapes, editorial magazine layouts, minimal with dramatic whitespace,
text integrated into the image area, etc.

Return the complete HTML file."""

    try:
        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=8192,
            system=TEMPLATE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        html = response.content[0].text.strip()

        # Strip markdown fences if present
        if html.startswith("```"):
            html = html.split("\n", 1)[1]
            if html.endswith("```"):
                html = html[:-3]
            html = html.strip()

        # Validate required placeholders
        required = ["{{HEADLINE}}", "{{IMAGE_PATH}}", "{{SUBTEXT}}", "{{CLIENT_NAME}}"]
        missing = [p for p in required if p not in html]
        if missing:
            logger.warning(f"Dynamic template missing placeholders: {missing} — patching...")
            if "{{IMAGE_PATH}}" not in html:
                html = html.replace(
                    "</body>",
                    '<img src="{{IMAGE_PATH}}" style="position:absolute;top:0;right:0;width:50%;height:100%;object-fit:cover;" alt="hero">\n</body>'
                )
            if "{{HEADLINE}}" not in html:
                html = html.replace(
                    "</body>",
                    '<h1 style="position:absolute;bottom:200px;left:64px;font-family:var(--font-headline);font-size:var(--font-headline-size);color:var(--color-text);">{{HEADLINE}}</h1>\n</body>'
                )
            if "{{SUBTEXT}}" not in html:
                html = html.replace(
                    "</body>",
                    '<p style="position:absolute;bottom:140px;left:64px;font-family:var(--font-subtext);color:var(--color-subtext);">{{SUBTEXT}}</p>\n</body>'
                )
            if "{{CLIENT_NAME}}" not in html:
                html = html.replace(
                    "</body>",
                    '<span style="position:absolute;bottom:40px;right:48px;font-size:14px;color:var(--client-color);text-transform:uppercase;letter-spacing:0.1em;opacity:0.7;">{{CLIENT_NAME}}</span>\n</body>'
                )

        logger.info(f"Dynamic template generated ({len(html)} chars)")
        return html

    except Exception as e:
        logger.error(f"Dynamic template generation failed: {e}")
        raise


def _pick_reference(
    refs: list, liked: list, template_hint: str,
    previous_templates: list[str] | None = None,
) -> str:
    """Pick an inspiration reference to base the layout on."""

    # If we have liked templates, 50% chance to use one
    if liked and random.random() < 0.5:
        entry = random.choice(liked)
        try:
            data = json.loads(entry["content"])
            return _format_reference(data, source="liked")
        except (json.JSONDecodeError, KeyError):
            pass

    # Otherwise pick from taste references, preferring variety
    if not refs:
        return "No inspiration references available. Create a clean, modern editorial layout."

    # Group refs by composition type
    by_comp = {}
    for ref in refs:
        try:
            data = json.loads(ref["content"])
            comp_type = data.get("composition", {}).get("template_match", "unknown")
            if comp_type not in by_comp:
                by_comp[comp_type] = []
            by_comp[comp_type].append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    # Pick a random composition type (weighted toward less-used ones for variety)
    if by_comp:
        # Inverse weighting: less common types get picked more
        types = list(by_comp.keys())
        counts = [len(by_comp[t]) for t in types]
        max_count = max(counts)
        weights = [max_count - c + 1 for c in counts]  # inverse

        chosen_type = random.choices(types, weights=weights, k=1)[0]
        chosen_ref = random.choice(by_comp[chosen_type])
        return _format_reference(chosen_ref, source=f"inspiration ({chosen_type})")

    # Fallback: random ref
    ref = random.choice(refs)
    try:
        data = json.loads(ref["content"])
        return _format_reference(data, source="random inspiration")
    except (json.JSONDecodeError, KeyError):
        return "Clean, modern editorial layout with generous whitespace."


def _format_reference(data: dict, source: str = "") -> str:
    """Format a taste reference into a description for the template generator."""
    parts = [f"Source: {source}"]

    # Composition
    comp = data.get("composition", {})
    if comp:
        parts.append(f"Composition: {comp.get('template_match', '?')} layout")
        if comp.get("text_position"):
            parts.append(f"  Text position: {comp['text_position']}")
        if comp.get("negative_space_pct"):
            parts.append(f"  Negative space: {comp['negative_space_pct']}%")
        if comp.get("visual_hierarchy"):
            parts.append(f"  Hierarchy: {comp['visual_hierarchy']}")

    # Layers (new format)
    layers = data.get("layers", [])
    if layers:
        parts.append("Layers (back to front):")
        for layer in layers:
            parts.append(
                f"  {layer.get('order', '?')}. [{layer.get('type', '?')}] "
                f"{layer.get('description', '')} — position: {layer.get('position', '?')}, "
                f"size: {layer.get('size', '?')}"
            )

    # Typography
    typo = data.get("typography", {})
    if isinstance(typo, list) and typo:
        parts.append("Typography:")
        for t in typo[:3]:
            parts.append(
                f"  {t.get('role', '?')}: {t.get('font_category', '?')} "
                f"{t.get('weight', '?')} — {t.get('case', '?')}, "
                f"tracking: {t.get('tracking', '?')}"
            )
    elif isinstance(typo, dict) and typo.get("font_category"):
        parts.append(f"Typography: {typo.get('font_category')} {typo.get('estimated_weight', '')}")

    # Colors
    colors = data.get("colors", {})
    if colors.get("temperature"):
        parts.append(f"Color temperature: {colors['temperature']}")
    if colors.get("palette_mood"):
        parts.append(f"Palette mood: {colors['palette_mood']}")
    palette = colors.get("palette", [])
    if palette:
        color_list = [f"{c.get('name', '?')} ({c.get('hex', '?')}) — {c.get('usage', '?')}" for c in palette[:5]]
        parts.append(f"Colors: {'; '.join(color_list)}")

    # Feeling
    feel = data.get("feeling", {})
    if feel.get("mood"):
        parts.append(f"Mood: {feel['mood']}")
    if feel.get("communicates"):
        parts.append(f"Communicates: {feel['communicates']}")

    # Rules
    rules = data.get("reusable_rules", [])
    if rules:
        parts.append("Design rules from this reference:")
        for r in rules[:3]:
            parts.append(f"  - {r}")

    return "\n".join(parts)


async def save_liked_template(
    brain: Brain,
    decisions: CreativeDecisions,
    template_html: str,
    concept_summary: str,
) -> None:
    """Save a liked template combo to the Brain for future reference."""
    liked_data = {
        "template_style": decisions.template,
        "font_headline": decisions.font_headline,
        "font_headline_weight": decisions.font_headline_weight,
        "color_bg": decisions.color_bg,
        "color_text": decisions.color_text,
        "color_accent": decisions.color_accent,
        "concept": concept_summary,
        "headline": decisions.headline,
    }

    brain.store(
        topic="liked_template",
        source="user_feedback",
        content=json.dumps(liked_data),
        client="ALL",
        summary=f"Liked: {decisions.template} with {decisions.font_headline}, concept: {concept_summary[:50]}",
        tags=["liked", decisions.template, decisions.font_headline],
    )
    logger.info(f"Saved liked template: {decisions.template} + {decisions.font_headline}")
