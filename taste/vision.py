"""
Claude Vision analysis — the heart of the taste engine.

Takes an inspiration image and returns a full structured breakdown:
typography, colors, composition, feeling, what makes it work, reusable rules.
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

1. TYPOGRAPHY — font category (geometric sans, humanist sans, transitional serif, slab serif,
   display, handwritten), estimated weight (100-900), estimated size in pt, letter-spacing/tracking,
   line-height, text case, hierarchy (what's biggest, what's secondary), what the typography communicates.

2. COLORS — exact hex codes for background, primary text, accent, and any other colors visible.
   Color temperature (warm/cool/neutral). Contrast level. Overall palette mood.

3. COMPOSITION — layout type (object-hero, text-dominant, split, full-bleed, grid, asymmetric).
   Object coverage percentage. Negative space percentage. Text position on the image.
   Text alignment. Approximate margins in pixels (assuming 1080x1080). Visual weight distribution.
   Grid feeling (editorial, centered, asymmetric, etc).

4. FEELING — mood, energy level (low/medium/high), sophistication level, brand impression.

5. WHAT MAKES IT WORK — 2-3 sentences explaining WHY this design is effective. Be specific.

6. REUSABLE RULES — 3-5 concrete, actionable design rules extracted from this image that can
   be applied to future designs.

Return ONLY valid JSON. No markdown, no explanation outside the JSON.

The JSON schema:
{
  "typography": {
    "font_category": "string",
    "estimated_weight": "string (e.g. 700-800)",
    "estimated_size_pt": number,
    "tracking": "string (e.g. tight, -0.02em)",
    "line_height": number,
    "case": "string (uppercase/lowercase/sentence/mixed)",
    "hierarchy": "string",
    "what_it_communicates": "string"
  },
  "colors": {
    "background": {"hex": "string", "name": "string"},
    "text_primary": {"hex": "string", "name": "string"},
    "accent": {"hex": "string", "name": "string"},
    "additional": [{"hex": "string", "name": "string", "usage": "string"}],
    "temperature": "string (warm/cool/neutral)",
    "contrast": "string (low/medium/high)",
    "palette_mood": "string"
  },
  "composition": {
    "template_match": "string (object-hero/text-dominant/split/full-bleed/grid/asymmetric)",
    "object_coverage_pct": number,
    "negative_space_pct": number,
    "text_position": "string (top-left/top-right/bottom-left/bottom-right/center/etc)",
    "text_alignment": "string (left/center/right)",
    "margin_px": number,
    "visual_weight_distribution": "string",
    "grid_feeling": "string"
  },
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

    Args:
        image_path: Path to the image file.
        context: Optional user caption/context (e.g. "love the typography here",
                 "this is for Somamed"). If provided, the analysis focuses on
                 what the user highlighted.

    Returns:
        Full breakdown dict with keys: typography, colors, composition,
        feeling, what_makes_it_work, reusable_rules.
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
    typo = analysis.get("typography", {})
    colors = analysis.get("colors", {})
    comp = analysis.get("composition", {})
    feel = analysis.get("feeling", {})
    rules = analysis.get("reusable_rules", [])

    lines = ["🔍 <b>Here's what I see in this image:</b>\n"]

    # Typography
    lines.append("📝 <b>Typography</b>")
    lines.append(f"  • {typo.get('font_category', '?')}, ~{typo.get('estimated_size_pt', '?')}pt, weight {typo.get('estimated_weight', '?')}")
    lines.append(f"  • {typo.get('case', '?')} case, tracking: {typo.get('tracking', '?')}")
    lines.append(f"  • Communicates: {typo.get('what_it_communicates', '?')}")

    # Colors
    bg = colors.get("background", {})
    txt = colors.get("text_primary", {})
    accent = colors.get("accent", {})
    lines.append(f"\n🎨 <b>Colors</b>")
    lines.append(f"  • Background: {bg.get('name', '?')} ({bg.get('hex', '?')})")
    lines.append(f"  • Text: {txt.get('name', '?')} ({txt.get('hex', '?')})")
    if accent.get("hex"):
        lines.append(f"  • Accent: {accent.get('name', '?')} ({accent.get('hex', '?')})")
    lines.append(f"  • Temperature: {colors.get('temperature', '?')}, contrast: {colors.get('contrast', '?')}")

    # Composition
    lines.append(f"\n📐 <b>Composition</b>")
    lines.append(f"  • Layout: {comp.get('template_match', '?')}")
    lines.append(f"  • Object coverage: ~{comp.get('object_coverage_pct', '?')}%, negative space: ~{comp.get('negative_space_pct', '?')}%")
    lines.append(f"  • Text position: {comp.get('text_position', '?')}, {comp.get('text_alignment', '?')} aligned")
    lines.append(f"  • Grid: {comp.get('grid_feeling', '?')}")

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
