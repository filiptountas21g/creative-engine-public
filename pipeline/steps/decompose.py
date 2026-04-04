"""
Step 0: Deep Decomposition — breaks a reference image into individual elements.

Returns an asset manifest: every visual element with position, type, sourcing
strategy, and (for icons/shapes) inline SVG code.

This is the foundation of the image copier — it tells the template generator
EXACTLY what to build and where, instead of "replicate this vaguely."
"""

import json
import logging

import anthropic
import config

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


DECOMPOSE_SYSTEM = """You are an expert UI/design reverse-engineer.

Given a reference image, you DECOMPOSE it into individual visual elements — like
disassembling a design file into its layers panel.

For EVERY element you can identify, return:
- What it is
- Its exact position on the canvas (as percentages)
- How to reproduce it (CSS, SVG, photo, or AI generation)

Be exhaustive. Miss nothing. A dashboard design might have 15-25 elements.
An arrow, a dot, a thin line — if you can see it, list it.

ICONS are critical. For every icon (gear, robot, arrow, chart, etc.), write
the COMPLETE inline SVG code that recreates it. Use simple paths, 24x24 viewBox,
stroke style matching what you see (line/filled/duotone). The SVG must be valid
and self-contained.

Return ONLY valid JSON. No markdown fences, no explanation."""


DECOMPOSE_SCHEMA = """{
  "canvas": {
    "aspect_ratio": "e.g. 1:1, 16:9, 4:5",
    "background_type": "solid | gradient | photo | complex",
    "background_css": "CSS value if achievable (e.g. linear-gradient(...) or #hex), null if photo/complex"
  },
  "elements": [
    {
      "id": "unique_snake_case_id",
      "type": "background | photo | text | icon | ui_widget | shape | decorative | overlay | logo_mark",
      "description": "What this element IS — be specific",
      "position": {
        "x": "left edge as % of canvas width (e.g. '5%')",
        "y": "top edge as % of canvas height (e.g. '10%')",
        "width": "element width as % of canvas (e.g. '40%')",
        "height": "element height as % of canvas (e.g. '30%')"
      },
      "z_index": 0,
      "sourcing": "css | inline_svg | ai_photo | stock_photo | html_css | text_placeholder",
      "css_snippet": "CSS to recreate (for css/html_css type) or null",
      "svg_code": "Complete <svg>...</svg> code (for inline_svg type) or null",
      "photo_prompt": "AI generation prompt (for ai_photo type) or stock search query (for stock_photo) or null",
      "text_placeholder": "Which placeholder: HEADLINE | SUBTEXT | CTA | CLIENT_NAME | null",
      "style_notes": "opacity, blend-mode, filter, border-radius, backdrop-filter, etc.",
      "children": ["ids of elements contained within this one"]
    }
  ],
  "color_palette": [
    {"hex": "#...", "role": "primary | secondary | accent | background | text", "name": "descriptive name"}
  ],
  "fonts": {
    "primary": {"category": "e.g. geometric sans", "weight": "e.g. 700", "google_fonts_suggestion": "e.g. Inter"},
    "secondary": {"category": "...", "weight": "...", "google_fonts_suggestion": "..."}
  },
  "assembly_order": ["list of element ids from back to front — the build order"],
  "recreation_notes": "2-3 sentences: what are the trickiest parts to get right?"
}"""


async def decompose_reference(image_b64: str, media_type: str = "image/jpeg") -> dict:
    """
    Deep decomposition of a reference image into an element-level asset manifest.

    Args:
        image_b64: Base64-encoded image
        media_type: MIME type (image/jpeg, image/png, image/webp)

    Returns:
        Asset manifest dict with elements, positions, sourcing strategies, SVGs
    """
    import base64 as _b64

    # Detect actual media type from magic bytes if possible
    try:
        raw = _b64.b64decode(image_b64[:32])
        if raw[:4] == b'RIFF' or b'WEBP' in raw[:12]:
            media_type = "image/webp"
        elif raw[:8] == b'\x89PNG\r\n\x1a\n':
            media_type = "image/png"
        elif raw[:3] == b'GIF':
            media_type = "image/gif"
    except Exception:
        pass

    user_prompt = f"""Decompose this reference image into individual elements.

I need to recreate this EXACTLY in HTML/CSS. Give me every single element with
its position, type, and how to source/build it.

CRITICAL RULES:
1. For every ICON you see (gear, arrow, chart bars, person silhouette icon, etc.),
   write the COMPLETE SVG code. Use viewBox="0 0 24 24", stroke="currentColor",
   stroke-width="1.5" or "2" depending on the style you see. Make it accurate.

2. For UI widgets (cards, pills, buttons, progress bars, timelines), describe the
   CSS needed — border-radius, backdrop-filter, background opacity, etc.

3. For photos/portraits, write a detailed AI generation prompt that would reproduce
   the subject, lighting, angle, and mood.

4. For backgrounds (gradients, solid colors), give the exact CSS value.

5. Positions are PERCENTAGES of the canvas. Be precise — "roughly center" is not
   acceptable. "x: 45%, y: 8%, width: 30%, height: 12%" IS.

6. Don't skip decorative elements — thin lines, dots, subtle overlays, border frames.

Return the JSON following this schema:
{DECOMPOSE_SCHEMA}"""

    try:
        response = _client.messages.create(
            model=config.SONNET_MODEL,
            max_tokens=8192,
            system=DECOMPOSE_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        }
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }],
        )

        text = response.content[0].text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        manifest = json.loads(text)
        logger.info(
            f"Decomposed into {len(manifest.get('elements', []))} elements "
            f"({sum(1 for e in manifest.get('elements', []) if e.get('type') == 'icon')} icons, "
            f"{sum(1 for e in manifest.get('elements', []) if e.get('svg_code'))} SVGs)"
        )
        return manifest

    except json.JSONDecodeError as e:
        logger.error(f"Decompose returned invalid JSON: {e}")
        logger.debug(f"Raw response: {text[:500]}")
        return {"elements": [], "error": str(e)}
    except Exception as e:
        logger.error(f"Decompose failed: {e}")
        return {"elements": [], "error": str(e)}


def format_manifest_for_template(manifest: dict) -> str:
    """
    Format the asset manifest into text that Opus can consume
    when generating the HTML template.

    This is injected into the template generation prompt so Opus
    knows exactly what to build and where.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ELEMENT MANIFEST — Build each of these in the HTML")
    lines.append("=" * 60)

    canvas = manifest.get("canvas", {})
    lines.append(f"\nCanvas: {canvas.get('aspect_ratio', '1:1')}")
    if canvas.get("background_css"):
        lines.append(f"Background CSS: {canvas['background_css']}")
    elif canvas.get("background_type"):
        lines.append(f"Background: {canvas['background_type']} (use {{{{IMAGE_1}}}} for photo/complex backgrounds)")

    # Color palette
    palette = manifest.get("color_palette", [])
    if palette:
        lines.append(f"\nColor Palette:")
        for c in palette:
            lines.append(f"  {c.get('hex', '?')} — {c.get('role', '?')}: {c.get('name', '')}")

    # Fonts
    fonts = manifest.get("fonts", {})
    if fonts.get("primary"):
        p = fonts["primary"]
        lines.append(f"\nPrimary font: {p.get('google_fonts_suggestion', '?')} ({p.get('category', '?')}, {p.get('weight', '?')})")
    if fonts.get("secondary"):
        s = fonts["secondary"]
        lines.append(f"Secondary font: {s.get('google_fonts_suggestion', '?')} ({s.get('category', '?')}, {s.get('weight', '?')})")

    # Elements — grouped by type
    elements = manifest.get("elements", [])
    assembly = manifest.get("assembly_order", [e["id"] for e in elements])

    # Track which IMAGE_N slot each photo gets
    photo_slot = 1
    photo_mapping = {}

    lines.append(f"\n{'—' * 40}")
    lines.append(f"ELEMENTS (build order: back → front)")
    lines.append(f"{'—' * 40}")

    for elem_id in assembly:
        elem = next((e for e in elements if e.get("id") == elem_id), None)
        if not elem:
            continue

        etype = elem.get("type", "?")
        pos = elem.get("position", {})
        pos_str = f"x:{pos.get('x','?')} y:{pos.get('y','?')} w:{pos.get('width','?')} h:{pos.get('height','?')}"

        lines.append(f"\n[{etype.upper()}] {elem['id']}")
        lines.append(f"  Description: {elem.get('description', '?')}")
        lines.append(f"  Position: {pos_str}")
        lines.append(f"  Z-index: {elem.get('z_index', '?')}")

        sourcing = elem.get("sourcing", "?")
        if sourcing == "inline_svg" and elem.get("svg_code"):
            lines.append(f"  → EMBED THIS SVG directly in the HTML:")
            lines.append(f"    {elem['svg_code']}")
        elif sourcing in ("ai_photo", "stock_photo"):
            slot = f"{{{{IMAGE_{photo_slot}}}}}"
            photo_mapping[elem_id] = photo_slot
            lines.append(f"  → Use {slot} (photo will be generated/sourced separately)")
            lines.append(f"    Photo description: {elem.get('photo_prompt', '?')}")
            photo_slot += 1
        elif sourcing in ("css", "html_css"):
            lines.append(f"  → Build with CSS/HTML:")
            if elem.get("css_snippet"):
                lines.append(f"    {elem['css_snippet']}")
        elif sourcing == "text_placeholder":
            placeholder = elem.get("text_placeholder", "HEADLINE")
            lines.append(f"  → Use {{{{{{{placeholder}}}}}}} placeholder")

        if elem.get("style_notes"):
            lines.append(f"  Style: {elem['style_notes']}")

        if elem.get("children"):
            lines.append(f"  Contains: {', '.join(elem['children'])}")

    # Summary
    notes = manifest.get("recreation_notes", "")
    if notes:
        lines.append(f"\n{'—' * 40}")
        lines.append(f"TRICKY PARTS: {notes}")

    lines.append(f"\n{'—' * 40}")
    lines.append(f"PHOTO SLOT MAPPING:")
    for eid, slot in photo_mapping.items():
        elem = next((e for e in elements if e.get("id") == eid), {})
        lines.append(f"  {{{{IMAGE_{slot}}}}} = {elem.get('description', eid)}")

    return "\n".join(lines)


def get_photo_prompts(manifest: dict) -> list[dict]:
    """
    Extract photo generation prompts from the manifest.

    Returns list of dicts with:
        slot: int (1, 2, 3...)
        prompt: str (generation/search prompt)
        sourcing: str ("ai_photo" or "stock_photo")
        description: str (what the photo is)
        is_background: bool (z_index <= 0 or type == "background")
    """
    photos = []
    slot = 1
    for elem in manifest.get("elements", []):
        if elem.get("sourcing") in ("ai_photo", "stock_photo"):
            photos.append({
                "slot": slot,
                "prompt": elem.get("photo_prompt", elem.get("description", "")),
                "sourcing": elem["sourcing"],
                "description": elem.get("description", ""),
                "is_background": elem.get("z_index", 1) <= 0 or elem.get("type") == "background",
            })
            slot += 1
    return photos
