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
  {{LOGO_PATH}} — OPTIONAL client logo (only include if has_logo is true). Place it small (40-60px) in a corner.

CRITICAL RULES:
- Every template MUST include <img src="{{IMAGE_PATH}}"> for the hero image
- Do NOT use background-image CSS for the hero
- Make each layout genuinely different — vary grid structures, text placement, image sizing, decorative elements
- Use the inspiration as a starting point, not a copy
- Be creative with layout: overlapping elements, asymmetric grids, text over image, sidebar layouts, diagonal cuts, etc.
- NEVER write actual text content into the HTML. ONLY use placeholders: {{HEADLINE}}, {{SUBTEXT}}, {{CTA}}, {{CLIENT_NAME}}
  For example, write: <h1>{{HEADLINE}}</h1>
  NEVER write: <h1>Clear thinking. Elevated results.</h1>
  The render engine will replace the placeholders with the actual text.
- NEVER use hardcoded hex colors (like #2A2A2A or #F2EDE4) in your CSS.
  ALWAYS use the CSS custom properties: var(--color-bg), var(--color-text), var(--color-accent), var(--color-subtext), var(--client-color).
  For example: color: var(--color-text)  NOT  color: #2A2A2A
  background: var(--color-bg)  NOT  background: #F5F0E8
  This ensures colors can be changed dynamically without re-generating the template.

Return ONLY the complete HTML file. No explanation, no markdown fences."""


async def generate_dynamic_template(
    decisions: CreativeDecisions,
    brain: Brain,
    previous_templates: list[str] | None = None,
    has_logo: bool = False,
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

{"Include a small logo (40-60px) in a tasteful position using: <img src=\"{{LOGO_PATH}}\" ...>. The client has a logo on file." if has_logo else "No client logo available — use {{CLIENT_NAME}} text label instead."}

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

        # First: replace any hardcoded text that should be placeholders
        # Opus sometimes writes the actual text instead of {{HEADLINE}} etc.
        _text_to_placeholder = {
            decisions.headline: "{{HEADLINE}}",
            decisions.subtext: "{{SUBTEXT}}",
            decisions.cta: "{{CTA}}",
        }
        for actual_text, placeholder in _text_to_placeholder.items():
            if actual_text and placeholder not in html and actual_text in html:
                logger.info(f"Replacing hardcoded '{actual_text[:40]}...' with {placeholder}")
                html = html.replace(actual_text, placeholder, 1)  # Replace first occurrence only

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

        # Post-process: replace any hardcoded hex colors with CSS variables
        # Opus often writes inline colors like `color: #2A2A2A` instead of `color: var(--color-text)`
        import re
        _color_to_var = {
            decisions.color_bg: "var(--color-bg)",
            decisions.color_text: "var(--color-text)",
            decisions.color_accent: "var(--color-accent)",
            decisions.color_subtext: "var(--color-subtext)",
        }
        hardcoded_count = 0
        for hex_color, css_var in _color_to_var.items():
            if not hex_color or len(hex_color) < 4:
                continue
            # Don't replace inside CSS variable definitions (--color-*: #xxx)
            # Only replace in property values and inline styles
            pattern = rf'(?<!--color-[a-z]+:\s*)(?<=[:;\s])({re.escape(hex_color)})(?=[;\s"\'\)])'
            matches = re.findall(pattern, html, flags=re.IGNORECASE)
            if matches:
                hardcoded_count += len(matches)
                html = re.sub(pattern, css_var, html, flags=re.IGNORECASE)

        if hardcoded_count > 0:
            logger.info(f"Replaced {hardcoded_count} hardcoded color values with CSS variables in template")

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
    client: str = "ALL",
    modifications: dict | None = None,
) -> None:
    """Save a liked template combo to the Brain for future reference.

    Args:
        modifications: Optional dict of changes to apply before saving,
            e.g. {"color_accent": "#FF6B00", "font_headline": "Syne"}
    """
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

    # Apply modifications before saving
    if modifications:
        liked_data.update(modifications)
        logger.info(f"Applied modifications to liked template: {modifications}")

    brain.store(
        topic="liked_template",
        source="user_feedback",
        content=json.dumps(liked_data),
        client=client,
        summary=f"Liked: {liked_data['template_style']} with {liked_data['font_headline']}, accent: {liked_data['color_accent']}, client: {client}",
        tags=["liked", liked_data["template_style"], liked_data["font_headline"], client],
    )
    logger.info(f"Saved liked template for {client}: {liked_data['template_style']} + {liked_data['font_headline']}")


async def save_client_preference(
    brain: Brain,
    client: str,
    preferences: dict,
) -> None:
    """Save client-specific design preferences (colors, fonts, rules).

    Args:
        preferences: e.g. {"accent_color": "#FF6B00", "brand_colors": ["#FF6B00", "#1A1A1A"],
                           "preferred_font": "Syne", "rules": ["always use orange accent"]}
    """
    brain.store(
        topic="client_preference",
        source="user_feedback",
        content=json.dumps(preferences),
        client=client,
        summary=f"Client preferences for {client}: {', '.join(f'{k}={v}' for k, v in list(preferences.items())[:3])}",
        tags=["client_pref", client],
    )
    logger.info(f"Saved client preference for {client}: {preferences}")


def get_client_preferences(brain: Brain, client: str) -> dict:
    """Get accumulated preferences for a specific client."""
    prefs = brain.query(topic="client_preference", client=client, limit=10)
    liked = brain.query(topic="liked_template", client=client, limit=10)

    merged = {}
    # Merge all preference entries (latest wins)
    for p in prefs:
        try:
            data = json.loads(p["content"])
            merged.update(data)
        except (json.JSONDecodeError, KeyError):
            continue

    # Extract patterns from liked templates for this client
    liked_accents = []
    liked_fonts = []
    liked_templates = []
    for l in liked:
        try:
            data = json.loads(l["content"])
            if data.get("color_accent"):
                liked_accents.append(data["color_accent"])
            if data.get("font_headline"):
                liked_fonts.append(data["font_headline"])
            if data.get("template_style"):
                liked_templates.append(data["template_style"])
        except (json.JSONDecodeError, KeyError):
            continue

    if liked_accents:
        merged["liked_accents"] = liked_accents
    if liked_fonts:
        merged["liked_fonts"] = liked_fonts
    if liked_templates:
        merged["liked_templates"] = liked_templates

    return merged
