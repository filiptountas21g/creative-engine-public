"""
Claude Vision analysis — the heart of the taste engine.

Takes an inspiration image and returns a full structured breakdown:
typography (multiple), colors (all), composition, layers, feeling,
what makes it work, reusable rules.
"""

import base64
import json
import logging
from pathlib import Path

import anthropic
from PIL import Image

import config

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Max dimension for Claude Vision (optimal quality/cost)
MAX_IMAGE_DIM = 1568

ANALYSIS_SYSTEM_PROMPT = """You are an expert design analyst for a creative advertising agency.

You analyze inspiration images and extract EVERY visual detail into a structured JSON format.
Your analysis is used to build a design system — it must be precise, specific, and actionable.

For each image, extract:

1. TYPOGRAPHY — designs often have MULTIPLE typefaces. Extract EACH one separately.
   For every distinct typeface you see (headline, subhead, body, label, date, CTA, etc.), create
   an entry with: role (what it's used for), font category, estimated weight, size, tracking, case,
   and what it communicates. A design with 3 different text styles = 3 typography entries.

2. COLORS — extract EVERY distinct color visible in the design.
   Do NOT simplify. A design with 6 colors should have 6 colors listed.
   For each color: exact hex code, descriptive name, and where it's used in the design.
   Also: overall temperature, contrast level, palette mood.

3. COMPOSITION — layout type, coverage percentages, text positions, margins, grid feeling.

4. LAYERS — describe the image as if you're explaining it to someone who will rebuild it.
   List every visual element from back to front (z-order):
   - What is each layer? (background color, photo, color block, text, icon, badge, gradient, etc.)
   - Where is it positioned? (top-left, center, spanning full width, etc.)
   - How big is it relative to the canvas?
   - How does it interact with other layers? (overlapping, adjacent, floating, anchored)
   This is the most important section — it should read like assembly instructions.

5. FEELING — mood, energy level, sophistication level, brand impression.

6. WHAT MAKES IT WORK — 2-3 sentences explaining WHY this design is effective. Be specific.

7. REUSABLE RULES — 3-5 concrete, actionable design rules from this image.

Return ONLY valid JSON. No markdown, no explanation outside the JSON.

The JSON schema:
{
  "typography": [
    {
      "role": "string (e.g. headline, subhead, body, label, date, CTA, caption)",
      "text_content": "string (the actual text shown)",
      "font_category": "string (geometric sans, humanist sans, transitional serif, slab serif, display, handwritten, monospace)",
      "estimated_weight": "string (e.g. 700-800)",
      "estimated_size_pt": number,
      "tracking": "string (e.g. tight, -0.02em, normal, wide)",
      "case": "string (uppercase/lowercase/sentence/mixed)",
      "what_it_communicates": "string"
    }
  ],
  "colors": {
    "palette": [
      {"hex": "string", "name": "string", "usage": "string (where in the design)"}
    ],
    "temperature": "string (warm/cool/neutral/mixed)",
    "contrast": "string (low/medium/high)",
    "palette_mood": "string"
  },
  "composition": {
    "template_match": "string (object-hero/text-dominant/split/full-bleed/grid/asymmetric/card-based)",
    "object_coverage_pct": number,
    "negative_space_pct": number,
    "text_position": "string",
    "text_alignment": "string (left/center/right)",
    "margin_px": number,
    "visual_weight_distribution": "string",
    "grid_feeling": "string"
  },
  "layers": [
    {
      "order": number,
      "type": "string (background/photo/color-block/text/icon/badge/gradient/shape/logo/overlay)",
      "description": "string (what it is)",
      "position": "string (where on the canvas)",
      "size": "string (relative size, e.g. full-width, ~30% of canvas, small pill)",
      "interaction": "string (how it relates to other layers)"
    }
  ],
  "feeling": {
    "mood": "string",
    "energy": "string (low/medium/high)",
    "sophistication": "string (low/medium/high)",
    "brand_impression": "string"
  },
  "what_makes_it_work": "string",
  "reusable_rules": ["string", "string", "string"]
}
"""


def _resize_image(image_path: str | Path) -> bytes:
    """Resize image to max dimension for Claude Vision and return JPEG bytes."""
    img = Image.open(image_path)

    # Convert RGBA/P to RGB for JPEG
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Resize if larger than max dimension
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        ratio = MAX_IMAGE_DIM / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        logger.info(f"Resized {w}x{h} → {new_size[0]}x{new_size[1]}")

    # Save to bytes
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _image_to_base64(image_path: str | Path) -> str:
    """Read image, resize, and return base64 string."""
    image_bytes = _resize_image(image_path)
    return base64.b64encode(image_bytes).decode("utf-8")


async def analyze_inspiration(
    image_path: str | Path,
    context: str = "",
) -> dict:
    """
    Analyze an inspiration image with Claude Vision.

    Returns:
        Full breakdown dict with keys: typography, colors, composition,
        layers, feeling, what_makes_it_work, reusable_rules.
    """
    image_b64 = _image_to_base64(image_path)

    user_message = "Analyze this design reference image and extract every visual detail."
    if context:
        user_message += f"\n\nThe user said about this image: \"{context}\"\nPay special attention to what they highlighted."

    try:
        response = _client.messages.create(
            model=config.VISION_MODEL,
            max_tokens=4096,
            system=ANALYSIS_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_message,
                        },
                    ],
                }
            ],
        )

        # Parse JSON from response
        raw_text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]  # remove first line
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

        analysis = json.loads(raw_text)
        logger.info(
            f"Analyzed image: template={analysis.get('composition', {}).get('template_match', '?')}, "
            f"mood={analysis.get('feeling', {}).get('mood', '?')}"
        )
        return analysis

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Vision response as JSON: {e}\nRaw: {raw_text[:500]}")
        raise ValueError(f"Claude Vision returned invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Vision analysis failed: {e}")
        raise


def format_analysis_for_telegram(analysis: dict) -> str:
    """Format a vision analysis into a human-readable Telegram message."""
    typo_list = analysis.get("typography", [])
    colors = analysis.get("colors", {})
    comp = analysis.get("composition", {})
    layers = analysis.get("layers", [])
    feel = analysis.get("feeling", {})
    rules = analysis.get("reusable_rules", [])

    # Handle old format where typography is a dict instead of list
    if isinstance(typo_list, dict):
        typo_list = [typo_list]

    lines = ["🔍 <b>Here's what I see in this image:</b>\n"]

    # Typography (multiple)
    lines.append("📝 <b>Typography</b>")
    for t in typo_list:
        role = t.get("role", "text")
        text_shown = t.get("text_content", "")
        cat = t.get("font_category", "?")
        weight = t.get("estimated_weight", "?")
        size = t.get("estimated_size_pt", "?")
        case = t.get("case", "?")
        tracking = t.get("tracking", "?")
        communicates = t.get("what_it_communicates", "")

        text_preview = f' "{text_shown[:30]}"' if text_shown else ""
        lines.append(f"  <b>{role}</b>{text_preview}")
        lines.append(f"    {cat}, ~{size}pt, weight {weight}, {case}, tracking: {tracking}")
        if communicates:
            lines.append(f"    → {communicates}")

    # Colors (all of them)
    palette = colors.get("palette", [])
    # Backwards compatibility: old format with background/text_primary/accent/additional
    if not palette:
        bg = colors.get("background", {})
        txt = colors.get("text_primary", {})
        accent = colors.get("accent", {})
        additional = colors.get("additional", [])
        if bg.get("hex"):
            palette.append({"hex": bg["hex"], "name": bg.get("name", "?"), "usage": "background"})
        if txt.get("hex"):
            palette.append({"hex": txt["hex"], "name": txt.get("name", "?"), "usage": "primary text"})
        if accent.get("hex") and accent["hex"].upper() not in ("#FFFFFF", "#000000"):
            palette.append({"hex": accent["hex"], "name": accent.get("name", "?"), "usage": "accent"})
        for extra in additional:
            palette.append(extra)

    lines.append(f"\n🎨 <b>Colors</b>")
    for c in palette:
        lines.append(f"  • {c.get('name', '?')} ({c.get('hex', '?')}) — {c.get('usage', '?')}")
    lines.append(f"  • Temperature: {colors.get('temperature', '?')}, contrast: {colors.get('contrast', '?')}")

    # Composition
    lines.append(f"\n📐 <b>Composition</b>")
    lines.append(f"  • Layout: {comp.get('template_match', '?')}")
    lines.append(f"  • Object coverage: ~{comp.get('object_coverage_pct', '?')}%, negative space: ~{comp.get('negative_space_pct', '?')}%")
    lines.append(f"  • Text position: {comp.get('text_position', '?')}, {comp.get('text_alignment', '?')} aligned")
    lines.append(f"  • Grid: {comp.get('grid_feeling', '?')}")

    # Layers (new!)
    if layers:
        lines.append(f"\n🧱 <b>Layers (back → front)</b>")
        for layer in sorted(layers, key=lambda l: l.get("order", 0)):
            ltype = layer.get("type", "?")
            desc = layer.get("description", "?")
            pos = layer.get("position", "?")
            size = layer.get("size", "?")
            interaction = layer.get("interaction", "")
            lines.append(f"  {layer.get('order', '?')}. <b>{ltype}</b>: {desc}")
            lines.append(f"     📍 {pos}, {size}")
            if interaction:
                lines.append(f"     ↔️ {interaction}")

    # Feeling
    lines.append(f"\n✨ <b>Feeling</b>")
    lines.append(f"  • Mood: {feel.get('mood', '?')}")
    lines.append(f"  • Energy: {feel.get('energy', '?')}, sophistication: {feel.get('sophistication', '?')}")
    lines.append(f"  • Brand impression: {feel.get('brand_impression', '?')}")

    # What makes it work
    lines.append(f"\n💡 <b>What makes it work</b>")
    lines.append(f"  {analysis.get('what_makes_it_work', '?')}")

    # Rules
    if rules:
        lines.append(f"\n📋 <b>Reusable rules</b>")
        for rule in rules:
            lines.append(f"  • {rule}")

    lines.append("\n<i>Reply to confirm, correct, or direct this analysis.</i>")

    return "\n".join(lines)
