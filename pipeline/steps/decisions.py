"""Step 6: Creative Decisions — Opus picks headline, font, colors, layout."""

import json
import logging

import anthropic

import config
from pipeline.types import (
    PipelineInput, CreativeConcept, CopyOptions, ImageResult,
    BrainContext, CreativeDecisions,
)
from pipeline.steps.font_pool import build_font_instruction, validate_font_weight, get_scout_font_names, _detect_greek

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

EDITOR_SYSTEM = """You are a senior creative director doing a final edit pass.

You make all the design decisions for a social media post:
1. Pick the BEST headline from 3 options (one sentence why)
2. Choose a Google Font for the headline — READ the font pool carefully
3. Choose a Google Font for the subtext (can be different category from headline)
4. Set font sizes, weights, tracking, line-height, case
5. Derive the color palette from the brand + concept
6. Choose the template layout
7. Set positioning and spacing

FONT PAIRING TIPS:
- Serif headline + sans-serif body = classic editorial look
- Display/condensed headline + humanist sans body = modern impact
- Geometric sans headline + serif body = tech-meets-tradition
- Don't pair two fonts from the same category unless they contrast in weight/size
- For bold statements: try condensed or display fonts at heavy weights
- For elegance: try serif or light-weight geometric sans

Available templates (VARY your choice — don't always pick the same one):
  object-hero — hero image with text overlay (default for single objects)
  text-dominant — headline is the hero, minimal/no image
  split — text left, image right
  full-bleed — image fills canvas, text on strip

IMPORTANT: If the brief mentions wanting something "different", "new style", or "another approach" —
you MUST change the template, font, AND color scheme significantly from what was described.
Don't keep picking the same template and font for every post. Be creative and vary your choices.

Return ONLY valid JSON:
{
  "headline": "chosen headline text",
  "headline_reason": "why this one",
  "font_headline": "Google Font name",
  "font_headline_weight": 800,
  "font_headline_size": 68,
  "font_headline_tracking": "-0.02em",
  "font_headline_line_height": 1.05,
  "font_headline_case": "uppercase",
  "font_subtext": "Google Font name",
  "font_subtext_weight": 400,
  "font_subtext_size": 18,
  "color_bg": "#hex",
  "color_text": "#hex",
  "color_accent": "#hex",
  "color_subtext": "#hex",
  "headline_margin_x": 64,
  "headline_margin_y": 64,
  "headline_max_width": "75%",
  "image_padding": 100,
  "remove_background": false,
  "template": "object-hero",
  "subtext": "chosen subtext",
  "cta": "chosen CTA"
}

IMPORTANT about remove_background:
  Set to true when the AI-generated image is a SINGLE OBJECT/PRODUCT/PERSON that should float
  on the template's background color (e.g. object-hero with colored bg, product shots).
  Set to false when the image is a full scene, landscape, or should keep its own background
  (e.g. full-bleed, atmospheric photos, scenes with environment)."""


def _format_client_prefs(prefs: dict | None, client: str) -> str:
    """Format client-specific preferences for the decisions prompt."""
    if not prefs:
        return ""
    lines = [f"⚡ CLIENT-SPECIFIC PREFERENCES FOR {client.upper()} (MUST follow these):"]
    if prefs.get("accent_color") or prefs.get("color_accent"):
        color = prefs.get("accent_color") or prefs.get("color_accent")
        lines.append(f"  ACCENT COLOR: {color} — use this as the accent color")
    if prefs.get("brand_colors"):
        lines.append(f"  BRAND COLORS: {', '.join(prefs['brand_colors'])}")
    if prefs.get("preferred_font") or prefs.get("font_headline"):
        font = prefs.get("preferred_font") or prefs.get("font_headline")
        lines.append(f"  PREFERRED FONT: {font}")
    if prefs.get("liked_accents"):
        lines.append(f"  Previously liked accents: {', '.join(prefs['liked_accents'][:5])}")
    if prefs.get("liked_fonts"):
        lines.append(f"  Previously liked fonts: {', '.join(prefs['liked_fonts'][:5])}")
    if prefs.get("liked_templates"):
        lines.append(f"  Previously liked templates: {', '.join(prefs['liked_templates'][:5])}")
    if prefs.get("rules"):
        for r in prefs["rules"]:
            lines.append(f"  RULE: {r}")
    return "\n".join(lines)


def _format_previous(previous: list[dict] | None) -> str:
    """Format previous decisions so Opus knows what to avoid repeating."""
    if not previous:
        return ""
    last = previous[-1]
    banned_template = last.get("template", "split")

    # Ban last 3 fonts for real variety
    banned_fonts = list(dict.fromkeys(p.get("font", "") for p in previous[-3:] if p.get("font")))

    lines = [
        "⚠️ MANDATORY VARIETY RULES:",
        f"  DO NOT use template '{banned_template}' — it was used in the last post.",
    ]
    if banned_fonts:
        lines.append(f"  DO NOT use these fonts (recently used): {', '.join(banned_fonts)}")
    lines.extend([
        f"  You MUST pick a different template from: object-hero, text-dominant, split, full-bleed",
        f"  (excluding '{banned_template}')",
    ])
    if len(previous) >= 2:
        used_templates = set(p.get("template") for p in previous[-3:])
        remaining = [t for t in ["object-hero", "text-dominant", "split", "full-bleed"] if t not in used_templates]
        if remaining:
            lines.append(f"  Suggested templates to try: {', '.join(remaining)}")
    return "\n".join(lines)


async def creative_decisions(
    input: PipelineInput,
    concept: CreativeConcept,
    copy: CopyOptions,
    image: ImageResult | None,
    brain_ctx: BrainContext,
    previous_decisions: list[dict] | None = None,
    client_preferences: dict | None = None,
) -> CreativeDecisions:
    """
    Make all final creative decisions — headline, fonts, colors, layout.

    This is where taste data has the most impact — font and color choices
    are informed by accumulated preferences.
    """
    taste = brain_ctx.taste_context

    # Build font instruction with category rotation and scout fonts
    headlines = copy.headlines[:3] if copy.headlines else ["N/A"]
    previous_fonts = [p.get("font", "") for p in (previous_decisions or []) if p.get("font")]
    # Scout fonts injected by orchestrator via client_preferences
    scout_fonts = (client_preferences or {}).get("scout_fonts", [])
    font_instruction = build_font_instruction(headlines, previous_fonts, scout_fonts)

    user_msg = f"""Make final creative decisions for this post.

Client: {input.client}
Brief: {input.brief}
Platform: {input.platform}

Visual concept: {concept.object}
Emotional direction: {concept.emotional_direction}
Background color from concept: {concept.background}

Three headline options:
  1. {headlines[0]}
  2. {headlines[1] if len(headlines) > 1 else 'N/A'}
  3. {headlines[2] if len(headlines) > 2 else 'N/A'}

Subtext: {copy.subtext}
CTA: {copy.cta}

Image generated with: {image.model_used if image else 'TBD — image generated after layout'}

{font_instruction}

TASTE PREFERENCES (MUST FOLLOW — these are designs the user explicitly approved):
  APPROVED fonts: {', '.join(taste.get('preferred_fonts', []))} — ONLY use these fonts unless none fit.
  APPROVED templates: {', '.join(taste.get('preferred_compositions', []))} — ONLY use these layouts.
  APPROVED colors: {', '.join(taste.get('preferred_colors', []))}
  BLACKLISTED (user hated these): {', '.join(taste.get('avoid', []))}

LIKED TEMPLATES (the user saved these as favorites — match this style):
{chr(10).join(f"  • {t.get('template_style','?')} | {t.get('font_headline','?')} {t.get('font_headline_weight','')} | bg:{t.get('color_bg','?')} accent:{t.get('color_accent','?')} | \"{t.get('headline','')[:40]}\"" for t in taste.get('liked_templates_raw', [])[:5])}

CRITICAL: Pick font + template + colors FROM the liked templates above. Do NOT invent new combinations the user hasn't approved.

Brand tone: {brain_ctx.brand.get('tone', 'professional')}

{f'FORCED TEMPLATE: {input.template_override}' if input.template_override else ''}

{_format_previous(previous_decisions)}

{_format_client_prefs(client_preferences, input.client)}

Return only the JSON with all decisions."""

    try:
        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=2048,
            system=EDITOR_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)

        # Validate font weights
        chosen_headline_font = data.get("font_headline", "Inter")
        chosen_headline_weight = data.get("font_headline_weight", 800)
        chosen_subtext_font = data.get("font_subtext", "DM Sans")
        chosen_subtext_weight = data.get("font_subtext_weight", 400)
        needs_greek = _detect_greek(data.get("headline", "") + data.get("subtext", ""))
        chosen_headline_weight = validate_font_weight(chosen_headline_font, chosen_headline_weight, needs_greek)
        chosen_subtext_weight = validate_font_weight(chosen_subtext_font, chosen_subtext_weight, needs_greek)

        decisions = CreativeDecisions(
            headline=data.get("headline", copy.headlines[0] if copy.headlines else ""),
            headline_reason=data.get("headline_reason", ""),
            font_headline=chosen_headline_font,
            font_headline_weight=chosen_headline_weight,
            font_headline_size=data.get("font_headline_size", 68),
            font_headline_tracking=data.get("font_headline_tracking", "-0.02em"),
            font_headline_line_height=data.get("font_headline_line_height", 1.05),
            font_headline_case=data.get("font_headline_case", "uppercase"),
            font_subtext=chosen_subtext_font,
            font_subtext_weight=chosen_subtext_weight,
            font_subtext_size=data.get("font_subtext_size", 18),
            color_bg=data.get("color_bg", concept.background),
            color_text=data.get("color_text", "#2A2A2A"),
            color_accent=data.get("color_accent", "#C4A77D"),
            color_subtext=data.get("color_subtext", "#6B6B6B"),
            headline_margin_x=data.get("headline_margin_x", 64),
            headline_margin_y=data.get("headline_margin_y", 64),
            headline_max_width=data.get("headline_max_width", "75%"),
            image_padding=data.get("image_padding", 100),
            remove_background=data.get("remove_background", False),
            template=input.template_override or data.get("template", "object-hero"),
            subtext=data.get("subtext", copy.subtext),
            cta=data.get("cta", copy.cta),
        )

        logger.info(
            f"Decisions: template={decisions.template}, "
            f"font={decisions.font_headline} {decisions.font_headline_weight}, "
            f"headline='{decisions.headline[:40]}...'"
        )
        return decisions

    except Exception as e:
        logger.error(f"Creative decisions failed: {e}")
        raise
