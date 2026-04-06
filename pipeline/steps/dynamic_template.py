"""Step 6b: Dynamic Template — generate a fresh HTML layout each time from inspiration."""

import json
import logging
import random
import time

import anthropic

import config
from brain.client import Brain
from brain.drive_client import ensure_subfolder, upload_html, list_html_files, download_text
from pipeline.types import CreativeDecisions

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

TEMPLATE_SYSTEM = """You are an expert HTML/CSS designer building a social media post template.

You will receive:
1. A reference image and/or description (composition, colors, typography, layers)
2. The creative decisions (headline, font, colors, template style)

TWO MODES — read the prompt carefully to know which one applies:

MODE A — "REFERENCE TO REPLICATE": The user chose a specific design and wants it reproduced.
  → COPY the layout structure AS CLOSELY AS POSSIBLE in HTML/CSS.
  → Same grid, same element positions, same visual treatment, same proportions.
  → Only substitute: brand colors, fonts, headline text, client logo.
  → Do NOT get creative. Do NOT simplify. Reproduce what you see.

MODE B — "INSPIRATION REFERENCE": Use it as creative direction, not a blueprint.
  → Create a UNIQUE layout inspired by the mood, composition style, and aesthetic.
  → You have creative freedom — vary the grid, text placement, decorative elements.

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
  {{IMAGE_1}} — primary hero image (MUST be an <img> tag, NOT CSS background-image)
  {{IMAGE_2}}, {{IMAGE_3}}, {{IMAGE_4}}, {{IMAGE_5}}, {{IMAGE_6}} — OPTIONAL additional images
  {{IMAGE_PATH}} — alias for {{IMAGE_1}} (both work)
  {{CTA}} — call to action text
  {{CLIENT_NAME}} — client name for footer label
  {{LOGO_PATH}} — OPTIONAL client logo (only include if has_logo is true). Place it small (40-60px) in a corner.

MULTIPLE IMAGES: If the layout needs multiple images (e.g. a grid of speakers, a photo collage,
  a background + foreground), use {{IMAGE_1}}, {{IMAGE_2}}, etc. The render engine will source
  a separate image for each slot. You can use up to 6 images. Example:
  <img src="{{IMAGE_1}}" alt="speaker 1">
  <img src="{{IMAGE_2}}" alt="speaker 2">
  Use multiple images when the reference has multiple photos or when the layout benefits from it.

IMAGE DISPLAY — CRITICAL for multi-photo layouts:
- Every image container MUST be large enough to show the full subject. If the reference shows
  portrait photos of people, the container needs enough height to show head + shoulders at minimum.
- ALWAYS use object-fit: cover; object-position: top center; on portrait/people photos so faces
  are visible and not cropped from the top.
- NEVER make image containers so small that photos get cut in half. If 3 photos sit in a row,
  each container should be at least 200px tall (on a 1080px canvas) — more if the reference shows them larger.
- Match the IMAGE SIZE from the reference proportionally. If photos take up 40% of the canvas height
  in the reference, they should take up ~40% in your reproduction.
- Set overflow: visible on image containers unless you specifically need clipping.

TEXT PROPORTIONALITY — CRITICAL (this is the #1 most common mistake):
- ONLY {{HEADLINE}} should use var(--font-headline-size). Everything else must be MUCH smaller.
- Names under photos: 13-15px, font-weight 600, var(--color-text). Like a photo caption.
- Roles/titles under photos: 10-12px, font-weight 400, uppercase, letter-spacing 0.05em, var(--color-subtext). Like a tiny label.
- Company names under photos: 10-12px, same as roles.
- These are CAPTIONS, not headlines. If the text under a photo is bigger than 16px, it is WRONG.
- The visual hierarchy must be: HEADLINE (large) >> photo names (small) >> roles/titles (tiny).
- All text must fit within its column without overflowing or getting cut off.
  Set max-width on text containers and use overflow: hidden; text-overflow: ellipsis; white-space: nowrap; for long text.

IMAGE RULES — read carefully:
- GRADIENT or ABSTRACT BACKGROUND: Do NOT write CSS gradients. Use <img src="{{IMAGE_1}}"> as a full-bleed background (position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; z-index:0). The AI generates the visual. All text/elements go on top (z-index:1+).
- SUBJECT PHOTO: Use <img src="{{IMAGE_1}}"> normally as the hero image in the layout.
- TEXT ON SOLID COLOR (no photo, no gradient): Do NOT include any {{IMAGE}} placeholder. Use only var(--color-bg) as background. This is valid.
- NEVER use CSS background-image, CSS gradients, or linear-gradient() for visuals — always use <img> tags with {{IMAGE_N}} placeholders so the AI can generate or source the image.
- In MODE B: make each layout genuinely different — asymmetric grids, overlapping elements, editorial compositions, diagonal cuts.
- In MODE A: match the reference precisely — do not simplify or genericise it.
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


def _logo_instruction(has_logo: bool) -> str:
    if has_logo:
        return 'Include a small logo (40-60px) in a tasteful position using: <img src="{{LOGO_PATH}}" ...>. The client has a logo on file.'
    return "No client logo available — use {{CLIENT_NAME}} text label instead."


async def generate_dynamic_template(
    decisions: CreativeDecisions,
    brain: Brain,
    previous_templates: list[str] | None = None,
    has_logo: bool = False,
    forced_reference: dict | None = None,
    client: str = "ALL",
    anti_repetition: str | None = None,
    canvas_format: str = "square",
    asset_manifest: str | None = None,
) -> str:
    """
    Generate a fresh HTML template based on an inspiration reference.

    If forced_reference is provided (e.g. from a user-sent inspiration image),
    use it directly instead of picking randomly from Brain.

    Returns the HTML string ready for injection.
    """
    if forced_reference:
        # User just sent this inspiration and said "make something like this"
        reference_text = _format_reference(forced_reference)
        logger.info("Using forced inspiration reference from user's latest image")
    else:
        # Try loading a saved template from Drive — adapt it instead of generating from scratch
        drive_templates = await _try_load_drive_template(client)
        if drive_templates:
            logger.info("Using saved Drive template as base — Opus will adapt it")
            return await _adapt_saved_template(drive_templates, decisions, has_logo)

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

    replication_instruction = ""
    if forced_reference:
        if asset_manifest:
            # ENHANCED MODE A — element-level manifest available
            replication_instruction = f"""
THIS IS MODE A (ENHANCED) — ELEMENT-LEVEL REPLICATION.
The user picked this specific design. You have a detailed ELEMENT MANIFEST below
that breaks down every single element with positions, SVG code, and CSS snippets.

YOUR JOB:
1. Build each element from the manifest at its specified position
2. For elements with svg_code: EMBED the SVG directly in the HTML — do not skip any icons
3. For elements with css_snippet: use that exact CSS
4. For photo slots: use the {{{{IMAGE_N}}}} placeholders as mapped in the manifest
5. For text: use {{{{HEADLINE}}}}, {{{{SUBTEXT}}}}, {{{{CTA}}}}, {{{{CLIENT_NAME}}}} placeholders
6. Match positions precisely — the manifest gives x/y/width/height as percentages
7. Every decorative element matters — thin lines, dots, arrows, shapes. Build them ALL.

IMAGE SIZING — CRITICAL:
- Photo containers MUST match the proportional size from the reference/manifest.
- If the manifest says a photo is 30% of canvas width and 35% of canvas height, make
  the container EXACTLY that size. Do NOT shrink photos to tiny thumbnails.
- Use object-fit: cover; object-position: top center; on ALL portrait/people photos
  so faces show fully and are not cropped from the top.
- NEVER let images get cut in half — the container must be tall enough to show the subject.

TEXT SIZING — CRITICAL (the #1 most common mistake):
- Names under photos: 13-15px max, font-weight 600. NOT 24px, NOT 30px, NOT bold headings.
- Roles/titles (e.g. "Founder - Company"): 10-12px max, uppercase, letter-spacing 0.05em.
- These are tiny photo CAPTIONS, not section headings. Look at the reference — they are small.
- Only {{{{HEADLINE}}}} should be large. Everything else must be proportionally tiny.
- Set max-width and overflow: hidden; text-overflow: ellipsis; on each text column so text doesn't overflow.

ADAPT ONLY:
- Replace colors with CSS variables (var(--color-bg), var(--color-text), etc.)
- Replace fonts with var(--font-headline), var(--font-subtext)
- Use the {{{{LOGO_PATH}}}} or {{{{CLIENT_NAME}}}} for branding

{asset_manifest}"""
        else:
            # STANDARD MODE A — no manifest, just reference image + text description
            replication_instruction = """
THIS IS MODE A — REPLICATE, DO NOT REIMAGINE.
The user picked this specific design. Your job is to reproduce the layout in HTML/CSS.

REPRODUCE EXACTLY:
- The grid structure and overall composition
- Where the text sits (top-left, bottom, centered, overlapping image, etc.)
- The spacing and proportions between elements
- Any decorative elements: geometric shapes, lines, badges, overlays, gradient backgrounds
- The visual hierarchy — what's dominant, what's secondary
- The general color treatment (dark bg = dark bg, gradient = gradient, minimal = minimal)

ONLY ADAPT:
- Replace colors with the client's brand colors (using CSS variables)
- Replace fonts with the chosen brand font
- Use {{HEADLINE}}, {{SUBTEXT}}, {{CTA}} placeholders for text content
- Use {{LOGO_PATH}} or {{CLIENT_NAME}} for the brand mark

MULTI-PHOTO: If the reference has multiple photos, use {{IMAGE_1}}, {{IMAGE_2}}, etc. — one per slot.
Never use divs or colored boxes as image placeholders."""

    # Canvas dimensions
    canvas_w, canvas_h = (1920, 1080) if canvas_format == "landscape" else (1080, 1080)
    canvas_label = f"{canvas_w}x{canvas_h}px ({'landscape 16:9' if canvas_format == 'landscape' else 'square 1:1'})"

    uniqueness_instruction = ""
    if not forced_reference:
        uniqueness_instruction = f"""
LAYOUT CREATIVITY — THIS IS CRITICAL:
The default "split" layout (text left, image right) and "centered text over image" are BANNED unless
the anti-repetition context explicitly allows them. Every post must feel like a different designer made it.

Pick ONE of these structural approaches and commit to it fully:
- FULL BLEED: image covers entire {canvas_w}x{canvas_h}, bold text overlaid with contrast treatment
- EDITORIAL GRID: asymmetric columns (30/70, 20/80), text at unexpected weight
- TEXT DOMINANT: large typographic statement fills most of the canvas, tiny image accent
- LAYERED: elements overlap — text sits partially on image, partially on color block
- DIAGONAL / ANGLED: content blocks at angles, not aligned to a grid
- BOTTOM HEAVY: image top 60%, all text stacked at bottom on solid color band
- COLLAGE: multiple image fragments arranged as a composition
- OVERSIZED HEADLINE: one word or phrase enormous, fills 60%+ of canvas

Do NOT produce a 50/50 vertical split with text on the left."""

    user_prompt = f"""Create an HTML template for this post.

CANVAS SIZE: {canvas_label} — the body must be exactly {canvas_w}px wide and {canvas_h}px tall.

{"REFERENCE TO REPLICATE (match this layout closely):" if forced_reference else "INSPIRATION REFERENCE (use as creative starting point):"}
{reference_text}

CREATIVE DECISIONS FOR THIS POST:
  Template style hint: {decisions.template}
  Headline: {decisions.headline}
  Font: {decisions.font_headline} ({decisions.font_headline_weight})
  Colors: bg={decisions.color_bg}, text={decisions.color_text}, accent={decisions.color_accent}
{replication_instruction}
The template style hint is just a suggestion — you can interpret it freely.
For example, "split" doesn't have to be a strict 50/50 split — it could be 30/70, or angled, or overlapping.
{avoid_text}

{_logo_instruction(has_logo)}
{uniqueness_instruction}
{anti_repetition or ""}

Return the complete HTML file."""

    try:
        # Build message content — include the actual inspiration image if available
        message_content = []

        # If we have the actual inspiration image, send it to Opus so it can SEE the reference
        reference_image_b64 = None
        if forced_reference and isinstance(forced_reference, dict):
            reference_image_b64 = forced_reference.get("_image_b64")

        if reference_image_b64:
            # Detect actual image format from magic bytes — don't assume jpeg
            import base64 as _b64
            raw = _b64.b64decode(reference_image_b64[:32])
            if raw[:4] == b'RIFF' or raw[:4] == b'WEBP' or b'WEBP' in raw[:12]:
                ref_media_type = "image/webp"
            elif raw[:8] == b'\x89PNG\r\n\x1a\n':
                ref_media_type = "image/png"
            elif raw[:3] == b'GIF':
                ref_media_type = "image/gif"
            else:
                ref_media_type = "image/jpeg"  # default
            logger.info(f"Reference image media type detected: {ref_media_type}")

            message_content.append({
                "type": "text",
                "text": "REFERENCE IMAGE (replicate this layout):"
            })
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": ref_media_type,
                    "data": reference_image_b64,
                }
            })
            logger.info("Sending actual inspiration image to Opus for template generation")

        message_content.append({"type": "text", "text": user_prompt})

        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=8192,
            system=TEMPLATE_SYSTEM,
            messages=[{"role": "user", "content": message_content}],
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
        has_any_image = "{{IMAGE_PATH}}" in html or "{{IMAGE_1}}" in html
        required = ["{{HEADLINE}}", "{{SUBTEXT}}", "{{CLIENT_NAME}}"]
        missing = [p for p in required if p not in html]
        if not has_any_image:
            logger.warning("No image placeholder found — patching with {{IMAGE_1}}")
            html = html.replace(
                "</body>",
                '<img src="{{IMAGE_1}}" style="position:absolute;top:0;right:0;width:50%;height:100%;object-fit:cover;" alt="hero">\n</body>'
            )
        if missing:
            logger.warning(f"Dynamic template missing placeholders: {missing} — patching...")
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
            # Replace hex colors in style contexts, but skip CSS variable definitions
            escaped = re.escape(hex_color)
            # Match color after : or ; or space, before ; or " or ' or )
            pattern = rf'(?<=[:;\s])({escaped})(?=[;\s"\'\)])'
            # First find all matches, then filter out ones in variable definitions
            for match in re.finditer(pattern, html, flags=re.IGNORECASE):
                start = match.start()
                # Check if this is inside a CSS variable definition (--color-xxx: #hex)
                prefix = html[max(0, start - 30):start]
                if re.search(r'--color-\w+\s*:\s*$', prefix):
                    continue  # Skip — this is the variable definition itself
                hardcoded_count += 1
            # Now do the replacement, skipping variable definitions
            def _replace_if_not_var_def(m):
                start = m.start()
                prefix = html[max(0, start - 30):start]
                if re.search(r'--color-\w+\s*:\s*$', prefix):
                    return m.group(0)  # Keep original
                return css_var
            html = re.sub(pattern, _replace_if_not_var_def, html, flags=re.IGNORECASE)

        if hardcoded_count > 0:
            logger.info(f"Replaced {hardcoded_count} hardcoded color values with CSS variables in template")

        logger.info(f"Dynamic template generated ({len(html)} chars)")
        return html

    except Exception as e:
        logger.error(f"Dynamic template generation failed: {e}")
        raise


FIX_TEMPLATE_SYSTEM = """You are an expert HTML/CSS designer fixing a social media post template.

You receive:
1. The current HTML template
2. A screenshot of how it renders (as a description of issues)
3. Specific issues to fix

Your job: return the FIXED HTML. Apply ALL the requested fixes.

Rules:
- Return ONLY the complete fixed HTML file. No explanation, no markdown fences.
- Keep the same overall layout/structure — only fix the specific issues mentioned.
- ALWAYS use CSS custom properties for colors: var(--color-bg), var(--color-text), var(--color-accent), var(--color-subtext), var(--client-color)
- NEVER hardcode hex colors in the CSS
- NEVER replace placeholders ({{HEADLINE}}, {{SUBTEXT}}, etc.) with actual text
- Preserve all existing placeholders exactly as they are
- Make the minimum changes needed to fix each issue
- If the fix says "add more margin" → adjust the specific CSS property
- If the fix says "improve contrast" → add a background overlay or adjust positioning
- If the fix says "text overlapping" → adjust z-index, positioning, or add padding"""


async def fix_template_from_critique(
    current_html: str,
    critique_text: str,
    decisions: CreativeDecisions,
    reference_image_b64: str | None = None,
) -> str:
    """
    Fix an HTML template based on visual critique feedback.

    Takes the current HTML and critique instructions, returns fixed HTML.
    Optionally receives the reference image so Opus can see what the user wants.
    """
    user_prompt = f"""Fix this HTML template based on the visual review.

CURRENT HTML:
```html
{current_html}
```

VISUAL REVIEW FEEDBACK:
{critique_text}

DESIGN DECISIONS (for reference):
  Headline: {decisions.headline}
  Font: {decisions.font_headline} ({decisions.font_headline_weight}), size {decisions.font_headline_size}px
  Colors: bg={decisions.color_bg}, text={decisions.color_text}, accent={decisions.color_accent}
  Template style: {decisions.template}

Apply ALL the fixes mentioned in the review. Return the complete fixed HTML."""

    try:
        # Build message content — include reference image if available
        message_content = []

        if reference_image_b64:
            import base64 as _b64
            try:
                raw = _b64.b64decode(reference_image_b64[:32])
                if raw[:4] == b'RIFF' or b'WEBP' in raw[:12]:
                    ref_media = "image/webp"
                elif raw[:8] == b'\x89PNG\r\n\x1a\n':
                    ref_media = "image/png"
                else:
                    ref_media = "image/jpeg"
            except Exception:
                ref_media = "image/jpeg"

            message_content.append({"type": "text", "text": "REFERENCE IMAGE (this is what the user wants the design to look like):"})
            message_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": ref_media, "data": reference_image_b64}
            })
            logger.info("Sending reference image to Opus for template fix")

        message_content.append({"type": "text", "text": user_prompt})

        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=8192,
            system=FIX_TEMPLATE_SYSTEM,
            messages=[{"role": "user", "content": message_content}],
        )

        html = response.content[0].text.strip()

        # Strip markdown fences if present
        if html.startswith("```"):
            html = html.split("\n", 1)[1]
            if html.endswith("```"):
                html = html[:-3]
            html = html.strip()

        # Same post-processing as generate
        _text_to_placeholder = {
            decisions.headline: "{{HEADLINE}}",
            decisions.subtext: "{{SUBTEXT}}",
            decisions.cta: "{{CTA}}",
        }
        for actual_text, placeholder in _text_to_placeholder.items():
            if actual_text and placeholder not in html and actual_text in html:
                logger.info(f"Fix: replacing hardcoded '{actual_text[:40]}...' with {placeholder}")
                html = html.replace(actual_text, placeholder, 1)

        # Validate placeholders still present — if Opus dropped them, PATCH them back in
        has_any_image = "{{IMAGE_PATH}}" in html or "{{IMAGE_1}}" in html
        required = ["{{HEADLINE}}", "{{SUBTEXT}}", "{{CLIENT_NAME}}"]
        missing = [p for p in required if p not in html]
        if missing:
            logger.warning(f"Fixed template missing placeholders: {missing} — patching them in")
            # Build a small block for each missing placeholder and inject before </body>
            patch_parts = []
            for placeholder in missing:
                if placeholder == "{{HEADLINE}}":
                    patch_parts.append(
                        '<h1 style="font-family:var(--font-headline);font-size:var(--font-headline-size);'
                        'font-weight:var(--font-headline-weight);color:var(--color-text);'
                        'margin:20px 40px 0;">{{HEADLINE}}</h1>'
                    )
                elif placeholder == "{{SUBTEXT}}":
                    patch_parts.append(
                        '<p style="font-size:14px;color:var(--color-subtext);'
                        'margin:8px 40px;">{{SUBTEXT}}</p>'
                    )
                elif placeholder == "{{CLIENT_NAME}}":
                    patch_parts.append(
                        '<div style="position:absolute;bottom:20px;left:40px;'
                        'font-size:11px;color:var(--color-subtext);letter-spacing:0.05em;'
                        'text-transform:uppercase;">{{CLIENT_NAME}}</div>'
                    )
            if patch_parts and "</body>" in html:
                patch_html = "\n".join(patch_parts)
                html = html.replace("</body>", f"{patch_html}\n</body>")
                logger.info(f"Patched {len(missing)} missing placeholders into template")

        # Post-process hardcoded colors
        import re
        _color_to_var = {
            decisions.color_bg: "var(--color-bg)",
            decisions.color_text: "var(--color-text)",
            decisions.color_accent: "var(--color-accent)",
            decisions.color_subtext: "var(--color-subtext)",
        }
        for hex_color, css_var in _color_to_var.items():
            if not hex_color or len(hex_color) < 4:
                continue
            escaped = re.escape(hex_color)
            pattern = rf'(?<=[:;\s])({escaped})(?=[;\s"\'\)])'
            def _replace_if_not_var_def_fix(m, _css_var=css_var):
                start = m.start()
                prefix = html[max(0, start - 30):start]
                if re.search(r'--color-\w+\s*:\s*$', prefix):
                    return m.group(0)
                return _css_var
            html = re.sub(pattern, _replace_if_not_var_def_fix, html, flags=re.IGNORECASE)

        logger.info(f"Fixed template generated ({len(html)} chars)")
        return html

    except Exception as e:
        logger.error(f"Template fix failed: {e}")
        # Return original HTML if fix fails
        return current_html


async def describe_element_for_generation(
    reference_image_b64: str,
    element_description: str,
) -> str:
    """
    Opus looks at the reference image and writes an AI image generation prompt
    for a specific element the user wants to add/recreate.

    Args:
        reference_image_b64: The inspiration image (base64)
        element_description: What the user wants, e.g. "the woman", "the background gradient"

    Returns:
        A detailed image generation prompt suitable for Flux/Ideogram
    """
    import base64 as _b64
    try:
        raw = _b64.b64decode(reference_image_b64[:32])
        if raw[:4] == b'RIFF' or b'WEBP' in raw[:12]:
            media = "image/webp"
        elif raw[:8] == b'\x89PNG\r\n\x1a\n':
            media = "image/png"
        else:
            media = "image/jpeg"
    except Exception:
        media = "image/jpeg"

    try:
        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=500,
            system=(
                "You are an expert at writing AI image generation prompts. "
                "The user will show you a reference image and describe which element they want recreated. "
                "Write a detailed prompt that an AI image generator (Flux/Ideogram) can use to produce that element. "
                "Include: subject, pose/angle, lighting, style, mood, colors, background treatment. "
                "The prompt should recreate the LOOK and FEEL of that element, not be a generic description. "
                "Return ONLY the prompt text, nothing else. Keep it under 200 words."
            ),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "REFERENCE IMAGE:"},
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": reference_image_b64}},
                    {"type": "text", "text": f"Write an AI image generation prompt for this element: {element_description}\n\nThe prompt should recreate this specific visual element as closely as possible."},
                ],
            }],
        )
        prompt = response.content[0].text.strip()
        logger.info(f"Opus wrote generation prompt for '{element_description[:40]}': {prompt[:100]}...")
        return prompt
    except Exception as e:
        logger.error(f"Failed to generate element prompt from reference: {e}")
        return element_description  # Fallback to the user's description


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


async def save_template_to_drive(client: str, template_html: str, decisions: CreativeDecisions) -> str | None:
    """Save a liked template's HTML to Google Drive under the client's subfolder.
    Returns the Drive file ID or None if Drive is not configured."""
    if not config.DRIVE_TEMPLATES_FOLDER_ID:
        logger.info("No DRIVE_TEMPLATES_FOLDER_ID — skipping Drive save")
        return None

    try:
        # Ensure client subfolder exists
        client_folder_id = await ensure_subfolder(config.DRIVE_TEMPLATES_FOLDER_ID, client)

        # Generate filename with timestamp and style hint
        ts = int(time.time())
        filename = f"{decisions.template}_{decisions.font_headline}_{ts}.html"

        file_id = await upload_html(client_folder_id, filename, template_html)
        logger.info(f"Saved template HTML to Drive: {client}/{filename} (id={file_id})")
        return file_id

    except Exception as e:
        logger.error(f"Failed to save template to Drive: {e}")
        return None


async def load_templates_from_drive(client: str, limit: int = 5) -> list[str]:
    """Load saved template HTML files from Drive for a client.
    Returns a list of HTML strings (most recent first)."""
    if not config.DRIVE_TEMPLATES_FOLDER_ID:
        return []

    try:
        # Find client subfolder
        client_folder_id = await ensure_subfolder(config.DRIVE_TEMPLATES_FOLDER_ID, client)
        files = await list_html_files(client_folder_id)

        templates = []
        for f in files[:limit]:
            try:
                html = await download_text(f["id"])
                templates.append(html)
                logger.info(f"Loaded template from Drive: {client}/{f['name']}")
            except Exception as e:
                logger.warning(f"Failed to download template {f['name']}: {e}")

        return templates

    except Exception as e:
        logger.error(f"Failed to load templates from Drive: {e}")
        return []


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


# ── Drive template helpers ───────────────────────────────────

async def _try_load_drive_template(client: str) -> str | None:
    """Try to load a saved template from Drive for this client.
    Returns the HTML string or None if no saved templates exist.
    Uses a 30% chance to encourage variety (70% generates fresh)."""
    if not config.DRIVE_TEMPLATES_FOLDER_ID:
        return None

    # Only use saved templates 30% of the time for variety
    if random.random() > 0.3:
        return None

    try:
        templates = await load_templates_from_drive(client, limit=5)
        if templates:
            chosen = random.choice(templates)
            logger.info(f"Picked saved Drive template for {client} ({len(chosen)} chars)")
            return chosen
    except Exception as e:
        logger.warning(f"Drive template load failed: {e}")

    return None


async def _adapt_saved_template(
    saved_html: str,
    decisions: CreativeDecisions,
    has_logo: bool = False,
) -> str:
    """Adapt a saved template to new content — Opus modifies the existing HTML
    instead of generating from scratch. Much cheaper in tokens."""

    adapt_prompt = f"""You have a saved HTML template that the user previously approved.
Adapt it for NEW content while keeping the same layout structure.

CHANGES TO MAKE:
- Headline: {decisions.headline}
- Subtext: {decisions.subtext}
- CTA: {decisions.cta}
- Font: {decisions.font_headline} ({decisions.font_headline_weight})
- Colors: bg={decisions.color_bg}, text={decisions.color_text}, accent={decisions.color_accent}
- {_logo_instruction(has_logo)}

RULES:
- Keep the EXACT same layout structure (grid, positioning, spacing)
- Only change text content, colors, and fonts
- Keep all {{IMAGE_1}}, {{IMAGE_2}} etc. placeholders
- Keep {{HEADLINE}}, {{SUBTEXT}}, {{CTA}}, {{CLIENT_NAME}} placeholders
- Use CSS variables (var(--color-bg), var(--font-headline), etc.) not hardcoded values
- Return ONLY the complete HTML file

SAVED TEMPLATE:
{saved_html}"""

    try:
        response = _client.messages.create(
            model=config.OPUS_MODEL,
            max_tokens=8192,
            system="You adapt HTML templates to new content. Keep the layout, change the content. Return only HTML.",
            messages=[{"role": "user", "content": adapt_prompt}],
        )

        html = response.content[0].text.strip()
        if html.startswith("```"):
            html = html.split("\n", 1)[1]
            if html.endswith("```"):
                html = html[:-3]
            html = html.strip()

        logger.info(f"Adapted saved template ({len(html)} chars)")
        return html

    except Exception as e:
        logger.error(f"Template adaptation failed: {e}")
        # Fall through — caller should generate fresh
        raise
