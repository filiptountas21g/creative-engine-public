"""
Template builder — generates HTML templates FROM accumulated taste data.

Analyzes taste references stored in Big Brain, finds layout/typography/color
patterns, and generates 4 HTML template files with CSS custom properties.

Templates can be regenerated anytime as more references are added.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import anthropic

import config
from brain.client import Brain

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

TEMPLATE_CATEGORIES = ["object-hero", "text-dominant", "split", "full-bleed"]

BUILDER_SYSTEM_PROMPT = """You are an expert HTML/CSS designer building social media post templates.

You receive accumulated design preferences (typography, colors, composition patterns) extracted
from inspiration images. Your job is to create a production-ready HTML template that embodies
these preferences.

The template MUST:
1. Be exactly 1080x1080px (LinkedIn/Instagram square)
2. Use CSS custom properties (variables) for ALL dynamic values — fonts, colors, sizes, positions
3. Load Google Fonts via <link> tag with {{FONT_URL}} placeholder
4. Have NO JavaScript
5. Be self-contained (inline CSS, no external stylesheets except Google Fonts)
6. Support Greek text (use unicode-range or fonts with greek subset)
7. Have clean, semantic HTML

CSS variables the template MUST define in :root:
  --font-headline, --font-headline-weight, --font-headline-size,
  --font-headline-tracking, --font-headline-line-height, --font-headline-case
  --font-subtext, --font-subtext-weight, --font-subtext-size
  --color-bg, --color-text, --color-accent, --color-subtext
  --headline-margin-x, --headline-margin-y, --headline-max-width
  --image-padding, --image-object-fit
  --client-color

Placeholders to use in the HTML (MANDATORY — the render engine replaces these):
  {{FONT_URL}} — Google Fonts <link> href (use in: <link href="{{FONT_URL}}" rel="stylesheet">)
  {{HEADLINE}} — main headline text (use in a heading element)
  {{SUBTEXT}} — supporting text (use in a paragraph element)
  {{IMAGE_PATH}} — path to hero image file (use in: <img src="{{IMAGE_PATH}}" ...> — MUST be an <img> tag, NOT a CSS background-image)
  {{CTA}} — call to action text
  {{CLIENT_NAME}} — client name for footer label

CRITICAL: Every template MUST include an <img src="{{IMAGE_PATH}}"> tag for the hero image.
Do NOT use background-image CSS for the hero — it won't work with the render pipeline.
The image should be styled with object-fit and proper sizing via CSS.

Return ONLY the complete HTML file. No explanation, no markdown fences."""


async def build_templates(brain: Brain) -> dict[str, Path]:
    """
    Generate all 4 HTML templates from accumulated taste data.

    Returns dict of template_name → file path.
    """
    # Gather taste data
    refs = brain.query(topic="taste_reference", limit=50)
    confirmed_typo = brain.query(topic="taste_typography", limit=20)
    confirmed_colors = brain.query(topic="taste_colors", limit=20)
    corrections = brain.query(topic="taste_correction", limit=15)
    rejected = brain.query(topic="taste_rejected", limit=10)

    if len(refs) < 3:
        raise ValueError(
            f"Need at least 3 taste references to build templates. "
            f"Currently have {len(refs)}. Feed more inspiration images!"
        )

    # Extract patterns
    taste_summary = _extract_patterns(refs, confirmed_typo, confirmed_colors, corrections, rejected)

    logger.info(f"Building templates from {len(refs)} references...")

    templates_dir = Path(config.TEMPLATES_DIR)
    templates_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for category in TEMPLATE_CATEGORIES:
        logger.info(f"Generating template: {category}")

        html = await _generate_template(category, taste_summary)

        out_path = templates_dir / f"{category}.html"
        out_path.write_text(html, encoding="utf-8")

        results[category] = out_path
        logger.info(f"Saved template: {out_path}")

    return results


async def _generate_template(category: str, taste_summary: dict) -> str:
    """Generate a single HTML template for a given category."""

    category_descriptions = {
        "object-hero": (
            "Hero image fills the frame with generous padding. "
            "Single object centered. Headline overlaid in a safe zone. "
            "The image IS the concept — text completes it."
        ),
        "text-dominant": (
            "The headline IS the hero — large, bold, commanding. "
            "Image is secondary (small, corner, or absent). "
            "Great for announcements, quotes, thought leadership."
        ),
        "split": (
            "Equal weight between text and image, side by side. "
            "Left side: headline + subtext. Right side: image. "
            "Editorial feel, structured, clean division."
        ),
        "full-bleed": (
            "Image fills the entire 1080x1080 canvas. "
            "Text sits on a semi-transparent strip or panel at the bottom. "
            "Atmospheric, mood-driven, immersive."
        ),
    }

    user_prompt = f"""Build an HTML template for category: {category}

Description: {category_descriptions.get(category, '')}

Design preferences from {taste_summary['total_refs']} analyzed inspiration images:

TYPOGRAPHY PATTERNS:
  Most common font types: {', '.join(taste_summary['top_fonts']) or 'no data yet'}
  Common weights: {', '.join(taste_summary['top_weights']) or '700-800'}
  Common tracking: {taste_summary.get('common_tracking', 'tight')}
  Common case: {taste_summary.get('common_case', 'mixed')}

COLOR PATTERNS:
  Temperature preference: {', '.join(taste_summary['top_temps']) or 'warm'}
  Common background hex: {taste_summary.get('common_bg_hex', '#F5F4F0')}
  Common text hex: {taste_summary.get('common_text_hex', '#2A2A2A')}
  Common accent hex: {taste_summary.get('common_accent_hex', '#C4A77D')}
  Palette mood: {taste_summary.get('palette_mood', 'quiet luxury')}

COMPOSITION PATTERNS:
  Preferred negative space: {taste_summary.get('avg_negative_space', '55-65')}%
  Common text position: {taste_summary.get('common_text_position', 'bottom-left')}
  Common margins: {taste_summary.get('common_margin', '64')}px
  Grid feeling: {taste_summary.get('grid_feeling', 'editorial')}

RULES (confirmed by the user):
{chr(10).join(f'  - {r}' for r in taste_summary.get('rules', ['generous negative space', 'one object one idea']))}

THINGS TO AVOID:
{chr(10).join(f'  - {a}' for a in taste_summary.get('avoid', ['busy backgrounds', 'aggressive colors']))}

The CSS custom property DEFAULT values should reflect these patterns.
The template structure should embody the composition preferences.
Make it production-quality — this will be screenshotted by Playwright at 1080x1080px."""

    response = _client.messages.create(
        model=config.OPUS_MODEL,
        max_tokens=8192,
        system=BUILDER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    html = response.content[0].text.strip()

    # Strip markdown fences if present
    if html.startswith("```"):
        html = html.split("\n", 1)[1]
        if html.endswith("```"):
            html = html[:-3]
        html = html.strip()

    # Validate required placeholders exist
    required = ["{{HEADLINE}}", "{{IMAGE_PATH}}", "{{SUBTEXT}}", "{{CLIENT_NAME}}"]
    missing = [p for p in required if p not in html]
    if missing:
        logger.warning(f"Template {category} missing placeholders: {missing} — patching...")
        # If IMAGE_PATH is missing, try to add it
        if "{{IMAGE_PATH}}" not in html:
            # Find a good spot to inject the image
            if "background-image" in html and "{{IMAGE_PATH}}" not in html:
                # AI used CSS background-image — convert to img tag approach
                html = html.replace("</body>", "")
                html += '\n<img src="{{IMAGE_PATH}}" style="position:absolute;top:0;right:0;width:50%;height:100%;object-fit:cover;" alt="hero">\n</body>'
            else:
                html = html.replace("</body>", '<img src="{{IMAGE_PATH}}" style="position:absolute;top:0;right:0;width:50%;height:100%;object-fit:cover;" alt="hero">\n</body>')

    return html


def _extract_patterns(
    refs: list, confirmed_typo: list, confirmed_colors: list,
    corrections: list, rejected: list,
) -> dict:
    """Extract design patterns from all taste data."""
    fonts = []
    weights = []
    temps = []
    bg_hexes = []
    text_hexes = []
    accent_hexes = []
    neg_spaces = []
    text_positions = []
    margins = []
    cases = []
    trackings = []
    moods = []
    rules = []
    avoid = []

    for ref in refs:
        try:
            data = json.loads(ref["content"])
            typo = data.get("typography", {})
            colors = data.get("colors", {})
            comp = data.get("composition", {})
            feel = data.get("feeling", {})

            # Typography: handle both list (new) and dict (old) format
            typo_entries = typo if isinstance(typo, list) else [typo] if isinstance(typo, dict) else []
            for t in typo_entries:
                if not isinstance(t, dict):
                    continue
                if t.get("font_category"):
                    fonts.append(t["font_category"])
                if t.get("estimated_weight"):
                    weights.append(str(t["estimated_weight"]))
                if t.get("case"):
                    cases.append(t["case"])
                if t.get("tracking"):
                    trackings.append(t["tracking"])

            # Colors: handle both palette array (new) and flat keys (old) format
            if colors.get("temperature"):
                temps.append(colors["temperature"])
            if colors.get("palette_mood"):
                moods.append(colors["palette_mood"])

            palette = colors.get("palette", [])
            if palette:
                # New format: palette is a list of {hex, name, usage}
                for c in palette:
                    usage = (c.get("usage") or "").lower()
                    hex_val = c.get("hex", "")
                    if not hex_val:
                        continue
                    if "background" in usage or "bg" in usage:
                        bg_hexes.append(hex_val)
                    elif "text" in usage or "headline" in usage or "body" in usage:
                        text_hexes.append(hex_val)
                    elif "accent" in usage or "highlight" in usage or "cta" in usage:
                        accent_hexes.append(hex_val)
            else:
                # Old format: background/text_primary/accent dicts
                bg = colors.get("background", {})
                if isinstance(bg, dict) and bg.get("hex"):
                    bg_hexes.append(bg["hex"])
                txt = colors.get("text_primary", {})
                if isinstance(txt, dict) and txt.get("hex"):
                    text_hexes.append(txt["hex"])
                acc = colors.get("accent", {})
                if isinstance(acc, dict) and acc.get("hex"):
                    accent_hexes.append(acc["hex"])

            if comp.get("negative_space_pct"):
                neg_spaces.append(comp["negative_space_pct"])
            if comp.get("text_position"):
                text_positions.append(comp["text_position"])
            if comp.get("margin_px"):
                margins.append(comp["margin_px"])
            for rule in data.get("reusable_rules", []):
                rules.append(rule)
        except (json.JSONDecodeError, KeyError):
            continue

    # Add confirmed data (higher weight)
    for entry in confirmed_typo:
        try:
            data = json.loads(entry["content"])
            # Handle both list (new) and dict (old) format
            typo_entries = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
            for t in typo_entries:
                if isinstance(t, dict) and t.get("font_category"):
                    fonts.extend([t["font_category"]] * 2)
        except (json.JSONDecodeError, KeyError):
            continue

    # Build avoid list
    for entry in rejected:
        try:
            data = json.loads(entry["content"])
            feel = data.get("feeling", {})
            if feel.get("mood"):
                avoid.append(f"{feel['mood']} mood")
        except (json.JSONDecodeError, KeyError):
            continue

    # Get most common values
    def _top(lst, n=3):
        return [v for v, _ in Counter(lst).most_common(n)]

    def _mode(lst, default=""):
        if not lst:
            return default
        return Counter(lst).most_common(1)[0][0]

    return {
        "total_refs": len(refs),
        "top_fonts": _top(fonts),
        "top_weights": _top(weights),
        "top_temps": _top(temps, 2),
        "common_bg_hex": _mode(bg_hexes, "#F5F4F0"),
        "common_text_hex": _mode(text_hexes, "#2A2A2A"),
        "common_accent_hex": _mode(accent_hexes, "#C4A77D"),
        "palette_mood": _mode(moods, "quiet luxury"),
        "avg_negative_space": f"{sum(neg_spaces) / len(neg_spaces):.0f}" if neg_spaces else "60",
        "common_text_position": _mode(text_positions, "bottom-left"),
        "common_margin": str(_mode(margins, 64)),
        "common_tracking": _mode(trackings, "tight"),
        "common_case": _mode(cases, "mixed"),
        "grid_feeling": "editorial",
        "rules": _top(rules, 5),
        "avoid": list(set(avoid))[:5],
    }


def format_templates_summary(results: dict[str, Path]) -> str:
    """Format template generation results for Telegram."""
    lines = ["📐 <b>Templates rebuilt!</b>\n"]
    for name, path in results.items():
        size_kb = path.stat().st_size / 1024
        lines.append(f"  • <b>{name}</b> — {size_kb:.1f} KB")
    lines.append(f"\nGenerated from your taste data. Say 'make a post for [client]' to test them.")
    return "\n".join(lines)
