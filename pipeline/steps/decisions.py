"""Step 6: Creative Decisions — Opus picks headline, font, colors, layout."""

import json
import logging

import anthropic

import config
from pipeline.types import (
    PipelineInput, CreativeConcept, CopyOptions, ImageResult,
    BrainContext, CreativeDecisions,
)

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

EDITOR_SYSTEM = """You are a senior creative director doing a final edit pass.

You make all the design decisions for a social media post:
1. Pick the BEST headline from 3 options (one sentence why)
2. Choose a Google Font for the headline (any Google Font — it will be downloaded dynamically)
3. Choose a Google Font for the subtext
4. Set font sizes, weights, tracking, line-height, case
5. Derive the color palette from the brand + concept
6. Choose the template layout
7. Set positioning and spacing

For GREEK TEXT, use fonts with greek subset support:
  Inter, DM Sans, Manrope, Noto Sans, Roboto, Open Sans, Lato,
  Source Sans 3, IBM Plex Sans, Nunito Sans

For LATIN TEXT, you can use any Google Font including:
  Fraunces, Playfair Display, Space Grotesk, Syne, Instrument Serif,
  Plus Jakarta Sans, DM Serif Display, Cormorant, Spectral

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
  "template": "object-hero",
  "subtext": "chosen subtext",
  "cta": "chosen CTA"
}"""


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
    banned_font = last.get("font", "")
    lines = [
        "⚠️ MANDATORY VARIETY RULES:",
        f"  DO NOT use template '{banned_template}' — it was used in the last post.",
        f"  DO NOT use font '{banned_font}' — it was used in the last post.",
        f"  You MUST pick a different template from: object-hero, text-dominant, split, full-bleed",
        f"  (excluding '{banned_template}')",
    ]
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
    image: ImageResult,
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

    user_msg = f"""Make final creative decisions for this post.

Client: {input.client}
Brief: {input.brief}
Platform: {input.platform}

Visual concept: {concept.object}
Emotional direction: {concept.emotional_direction}
Background color from concept: {concept.background}

Three headline options:
  1. {copy.headlines[0] if len(copy.headlines) > 0 else 'N/A'}
  2. {copy.headlines[1] if len(copy.headlines) > 1 else 'N/A'}
  3. {copy.headlines[2] if len(copy.headlines) > 2 else 'N/A'}

Subtext: {copy.subtext}
CTA: {copy.cta}

Image generated with: {image.model_used}

TASTE PREFERENCES:
  Preferred fonts: {', '.join(taste.get('preferred_fonts', []))}
  Preferred color temperatures: {', '.join(taste.get('preferred_color_temperatures', []))}
  Preferred compositions: {', '.join(taste.get('preferred_compositions', []))}
  Preferred moods: {', '.join(taste.get('preferred_moods', []))}
  Confirmed rules: {', '.join(taste.get('confirmed_rules', [])[:5])}
  Avoid: {', '.join(taste.get('avoid', []))}

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

        decisions = CreativeDecisions(
            headline=data.get("headline", copy.headlines[0] if copy.headlines else ""),
            headline_reason=data.get("headline_reason", ""),
            font_headline=data.get("font_headline", "Inter"),
            font_headline_weight=data.get("font_headline_weight", 800),
            font_headline_size=data.get("font_headline_size", 68),
            font_headline_tracking=data.get("font_headline_tracking", "-0.02em"),
            font_headline_line_height=data.get("font_headline_line_height", 1.05),
            font_headline_case=data.get("font_headline_case", "uppercase"),
            font_subtext=data.get("font_subtext", "DM Sans"),
            font_subtext_weight=data.get("font_subtext_weight", 400),
            font_subtext_size=data.get("font_subtext_size", 18),
            color_bg=data.get("color_bg", concept.background),
            color_text=data.get("color_text", "#2A2A2A"),
            color_accent=data.get("color_accent", "#C4A77D"),
            color_subtext=data.get("color_subtext", "#6B6B6B"),
            headline_margin_x=data.get("headline_margin_x", 64),
            headline_margin_y=data.get("headline_margin_y", 64),
            headline_max_width=data.get("headline_max_width", "75%"),
            image_padding=data.get("image_padding", 100),
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
