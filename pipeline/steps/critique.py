"""Visual Critique — renders template to PNG, Vision reviews it, returns fixes.

The render → critique → fix loop:
1. Playwright renders current HTML to a temporary PNG
2. Claude Vision analyzes the PNG against the intended design decisions
3. Returns specific, actionable critique points
4. Opus uses critique to fix the HTML
"""

import base64
import logging
from pathlib import Path

import anthropic

import config
from pipeline.types import CreativeDecisions

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


CRITIQUE_SYSTEM = """You are a senior art director reviewing a rendered social media post.

You receive:
1. The rendered PNG of the post (what the audience will see)
2. The intended design decisions (what the designer wanted)

Your job: find SPECIFIC visual problems that hurt the design quality.

Focus on these categories:
1. TEXT READABILITY — is the headline readable? Is there enough contrast with the background?
   Does text overlap the image in a way that makes it hard to read?
2. SPACING & BALANCE — are margins consistent? Is there enough breathing room?
   Does anything feel cramped or too far away?
3. HIERARCHY — is the headline the dominant element? Does the eye flow correctly?
   Headline → Image → Subtext → CTA → Client name
4. IMAGE PLACEMENT — is the hero image well-positioned? Does it feel intentional?
   Is it too small, too large, or awkwardly cropped?
5. OVERALL POLISH — does this look like a professional agency post or a template?
   Would you show this to a client?
6. LAYOUT ISSUES — are elements overlapping badly? Is anything cut off or hidden?
   Are there empty areas that feel unintentional?
7. COLOR HARMONY — do the colors work together? Does the accent color pop appropriately?
8. COLOR ACCURACY — CRITICAL: Compare the ACTUAL colors you see in the rendered image against
   the INTENDED colors from the design decisions. If the background is supposed to be #FFFFFF (white)
   but you see a dark blue background, that is a CRITICAL issue. Same for text color, accent color.
   The rendered output MUST match the intended color values. Any mismatch is a critical bug.

Be SPECIFIC and ACTIONABLE. Don't say "improve spacing" — say "the headline needs 20px more top margin to breathe."
Don't say "fix colors" — say "the headline text (#2A2A2A) doesn't have enough contrast against the image behind it — add a semi-transparent overlay or move text to the solid background area."

Return ONLY valid JSON:
{
  "score": 1-10 (10 = perfect, ready to send to client),
  "issues": [
    {
      "severity": "critical" | "major" | "minor",
      "category": "text_readability" | "spacing" | "hierarchy" | "image" | "polish" | "layout" | "color",
      "problem": "specific description of what's wrong",
      "fix": "specific CSS/HTML fix instruction"
    }
  ],
  "what_works": "1-2 sentences about what's good about this design",
  "overall": "1 sentence summary of the most important thing to fix"
}

If the design scores 8+ with no critical or major issues, return an empty issues array — it's good enough."""


async def critique_render(
    rendered_png_path: str,
    decisions: CreativeDecisions,
    template_html: str,
    iteration: int = 1,
    forced_reference: dict | None = None,
    taste_context: str | None = None,
    client_context: str | None = None,
) -> dict:
    """
    Have Claude Vision critique a rendered post.

    Returns dict with score, issues, what_works, overall.
    """
    # Load the rendered PNG as base64
    img_path = Path(rendered_png_path)
    if not img_path.exists():
        logger.error(f"Rendered PNG not found: {rendered_png_path}")
        return {"score": 5, "issues": [], "what_works": "", "overall": "Could not load image for critique"}

    img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    # Build reference context if we're replicating a specific design
    reference_context = ""
    if forced_reference:
        from pipeline.steps.dynamic_template import _format_reference
        ref_text = _format_reference(forced_reference, source="user's inspiration")
        reference_context = (
            f"\n\nIMPORTANT — REFERENCE LAYOUT TO MATCH:\n"
            f"The user asked to replicate a specific inspiration image. "
            f"The rendered post MUST match this layout structure:\n{ref_text}\n\n"
            f"Judge primarily on how well it matches the REFERENCE LAYOUT — "
            f"same grid structure, same text positioning, same visual hierarchy. "
            f"Do NOT suggest changing the layout to something different from the reference."
        )

    # Build taste context — lightweight reference, NOT a hard constraint
    taste_section = ""
    if taste_context:
        taste_section = (
            f"\n\nDESIGNER'S SAVED FAVORITES (for context only — do NOT force the design to match these):\n"
            f"{taste_context}\n"
            f"This is background info. Focus your critique on VISUAL QUALITY (readability, spacing, hierarchy, "
            f"color harmony) — NOT on whether the design matches past favorites. Variety is good."
        )

    # Build client brand context
    client_section = ""
    if client_context:
        client_section = (
            f"\n\nCLIENT BRAND CONTEXT:\n"
            f"{client_context}\n"
            f"The design must respect the client's brand identity."
        )

    content = [
        {
            "type": "text",
            "text": (
                f"Review this rendered social media post (iteration {iteration}).\n\n"
                f"INTENDED DESIGN:\n"
                f"  Headline: \"{decisions.headline}\"\n"
                f"  Subtext: \"{decisions.subtext}\"\n"
                f"  CTA: \"{decisions.cta}\"\n"
                f"  Font: {decisions.font_headline} ({decisions.font_headline_weight}), size {decisions.font_headline_size}px\n"
                f"  Colors: bg={decisions.color_bg}, text={decisions.color_text}, accent={decisions.color_accent}\n"
                f"  Template style: {decisions.template}\n"
                f"  Image padding: {decisions.image_padding}px\n"
                f"  Headline position: margins ({decisions.headline_margin_x}px, {decisions.headline_margin_y}px)\n"
                f"{reference_context}"
                f"{taste_section}"
                f"{client_section}\n\n"
                f"Look at the rendered result and tell me what needs fixing.\n"
                f"Be harsh but specific — this needs to look like it came from a top design agency."
            ),
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        },
    ]

    try:
        response = _client.messages.create(
            model=config.VISION_MODEL,
            max_tokens=1500,
            system=CRITIQUE_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        import json
        critique = json.loads(raw)

        score = critique.get("score", 5)
        issues = critique.get("issues", [])
        critical_count = sum(1 for i in issues if i.get("severity") == "critical")
        major_count = sum(1 for i in issues if i.get("severity") == "major")
        minor_count = sum(1 for i in issues if i.get("severity") == "minor")

        logger.info(
            f"Critique (iter {iteration}): score={score}/10, "
            f"issues={len(issues)} (critical={critical_count}, major={major_count}, minor={minor_count})"
        )
        if critique.get("overall"):
            logger.info(f"  → {critique['overall']}")

        return critique

    except Exception as e:
        logger.error(f"Critique failed: {e}")
        return {"score": 5, "issues": [], "what_works": "", "overall": f"Critique error: {e}"}


def needs_revision(critique: dict) -> bool:
    """Check if the critique warrants a revision pass."""
    score = critique.get("score", 5)
    issues = critique.get("issues", [])

    # Score 8+ with no critical/major = good enough
    if score >= 8:
        critical_or_major = [i for i in issues if i.get("severity") in ("critical", "major")]
        if not critical_or_major:
            return False

    # Any critical issue = must fix
    if any(i.get("severity") == "critical" for i in issues):
        return True

    # Score below 6 = must fix
    if score < 6:
        return True

    # 2+ major issues = should fix
    major_count = sum(1 for i in issues if i.get("severity") == "major")
    if major_count >= 2:
        return True

    return False


def format_critique_for_fix(critique: dict) -> str:
    """Format critique issues into instructions for Opus to fix the template."""
    issues = critique.get("issues", [])
    if not issues:
        return ""

    lines = ["FIX THESE ISSUES in the HTML/CSS:\n"]

    # Critical first, then major, then minor
    for severity in ("critical", "major", "minor"):
        severity_issues = [i for i in issues if i.get("severity") == severity]
        for issue in severity_issues:
            lines.append(
                f"[{severity.upper()}] {issue.get('category', '?')}: "
                f"{issue.get('problem', '?')}\n"
                f"  → Fix: {issue.get('fix', '?')}\n"
            )

    lines.append(f"\nOVERALL: {critique.get('overall', '')}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# EDIT VERIFICATION — did the user's feedback get applied?
# ══════════════════════════════════════════════════════════════

EDIT_CHECK_SYSTEM = """You are a visual QA checker for a social media post editor.

The user asked for a specific change to be made to a design. Your ONLY job:
did the change actually happen?

You are NOT reviewing the design quality, aesthetics, or anything else.
You are ONLY checking: was the user's specific request applied?

Return ONLY valid JSON:
{
  "applied": true | false,
  "confidence": 1-10 (10 = definitely applied, 1 = definitely not),
  "what_i_see": "1 sentence describing what the render actually shows regarding the requested change",
  "fix_instruction": "if not applied: specific CSS/HTML instruction to fix it. if applied: empty string"
}"""


async def check_edit_applied(
    rendered_png_path: str,
    user_feedback: str,
) -> dict:
    """
    Check if a user's edit feedback was actually applied in the rendered output.

    Only checks the specific thing the user asked for — no general design review.
    Returns dict with applied (bool), confidence, what_i_see, fix_instruction.
    """
    import json

    img_path = Path(rendered_png_path)
    if not img_path.exists():
        return {"applied": False, "confidence": 1, "what_i_see": "Could not load image", "fix_instruction": ""}

    img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    content = [
        {
            "type": "text",
            "text": (
                f"The user asked for this change:\n\n"
                f"\"{user_feedback}\"\n\n"
                f"Look at the rendered result below. Was this change applied?"
            ),
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        },
    ]

    try:
        response = _client.messages.create(
            model=config.VISION_MODEL,
            max_tokens=500,
            system=EDIT_CHECK_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        result = json.loads(raw)
        applied = result.get("applied", False)
        confidence = result.get("confidence", 5)
        logger.info(
            f"Edit check: applied={applied}, confidence={confidence}/10 — {result.get('what_i_see', '')}"
        )
        return result

    except Exception as e:
        logger.error(f"Edit check failed: {e}")
        # If check fails, don't block — assume it's fine
        return {"applied": True, "confidence": 5, "what_i_see": f"Check error: {e}", "fix_instruction": ""}


# ══════════════════════════════════════════════════════════════
# COPY-MODE COMPARISON — visual diff, not aesthetic judgment
# ══════════════════════════════════════════════════════════════

COMPARISON_SYSTEM = """You are a visual QA engineer comparing a rendered HTML reproduction against its reference design.

You receive TWO images:
- IMAGE A: The original reference design (the target layout to replicate)
- IMAGE B: The rendered reproduction (what was produced)

CRITICAL: The reproduction has DIFFERENT text content, DIFFERENT brand colors, and a DIFFERENT image
than the reference. This is expected — we are replicating the LAYOUT and STRUCTURE, not the content.

You are comparing STRUCTURE ONLY:
- Where elements are positioned (top-left, center, bottom-right, etc.)
- How many text blocks / image areas exist
- The visual hierarchy (what's big, what's small)
- Spacing and margins between elements
- Overall composition (split layout, full-bleed, text-dominant, etc.)

DO NOT penalize for:
- Different text content (headlines, subtext, CTA will always differ)
- Different colors (brand colors are intentionally different)
- Different images (a different photo/graphic is expected)
- Different font choices (fonts are chosen per-brand)
- Different logo

DO penalize for:
1. WRONG LAYOUT — e.g. reference has text on the right but render has it centered
2. MISSING STRUCTURAL ELEMENTS — e.g. reference has a subtitle area but render doesn't
3. EXTRA STRUCTURAL ELEMENTS — e.g. render has elements not in the reference layout
4. WRONG PROPORTIONS — e.g. image takes 30% in reference but 80% in render
5. WRONG HIERARCHY — e.g. headline is small when it should dominate
6. WRONG SPACING — margins or padding significantly off from reference

Return ONLY valid JSON:
{
  "similarity_pct": 0-100,
  "differences": [
    {
      "category": "wrong_layout | missing_element | extra_element | wrong_proportions | wrong_hierarchy | wrong_spacing",
      "what": "the specific structural element",
      "in_reference": "where/how it appears in the reference LAYOUT",
      "in_render": "where/how it appears in the render (or 'absent')",
      "fix": "specific CSS/HTML layout fix instruction"
    }
  ],
  "summary": "1 sentence — the single most important STRUCTURAL difference to fix"
}

If the LAYOUT is similar (similarity 75+), return an empty differences array.
A render with the right layout but different content/colors = 90-100% similar."""


async def compare_to_reference(
    rendered_png_path: str,
    reference_image_b64: str,
    iteration: int = 1,
    canvas_format: str = "square",
) -> dict:
    """
    Compare a rendered output to the reference image. Pure visual diff.
    No aesthetic judgment — just finding differences.
    """
    import json

    # Load rendered PNG
    rendered_path = Path(rendered_png_path)
    rendered_b64 = base64.b64encode(rendered_path.read_bytes()).decode("utf-8")

    # Detect reference media type
    try:
        raw = base64.b64decode(reference_image_b64[:32])
        if raw[:4] == b'RIFF' or b'WEBP' in raw[:12]:
            ref_media = "image/webp"
        elif raw[:8] == b'\x89PNG\r\n\x1a\n':
            ref_media = "image/png"
        else:
            ref_media = "image/jpeg"
    except Exception:
        ref_media = "image/jpeg"

    format_note = ""
    if canvas_format == "landscape":
        format_note = (
            "\n\nIMPORTANT: The reproduction was intentionally rendered in LANDSCAPE format (16:9) "
            "while the reference is square. Do NOT penalize the aspect ratio difference. "
            "Focus on whether the same ELEMENTS exist and their RELATIVE positions are correct. "
            "The wider canvas means elements may be spread out more — this is expected."
        )

    message_content = [
        {"type": "text", "text": "IMAGE A — THE REFERENCE DESIGN (this is the target to match):"},
        {"type": "image", "source": {"type": "base64", "media_type": ref_media, "data": reference_image_b64}},
        {"type": "text", "text": f"IMAGE B — YOUR RENDERED REPRODUCTION (iteration {iteration}):"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": rendered_b64}},
        {"type": "text", "text": f"List every difference between IMAGE A and IMAGE B. Be exhaustive but focus on structural differences, not minor pixel variations.{format_note}"},
    ]

    try:
        response = _client.messages.create(
            model=config.VISION_MODEL,
            max_tokens=1500,
            system=COMPARISON_SYSTEM,
            messages=[{"role": "user", "content": message_content}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
        n_diffs = len(result.get("differences", []))
        sim = result.get("similarity_pct", 0)
        logger.info(f"Comparison (iter {iteration}): {sim}% similar, {n_diffs} differences")
        if result.get("summary"):
            logger.info(f"  → {result['summary']}")
        return result

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return {"similarity_pct": 50, "differences": [], "summary": f"Comparison error: {e}"}


def needs_copy_revision(comparison: dict) -> bool:
    """Check if a copy-mode render needs revision based on structural comparison."""
    sim = comparison.get("similarity_pct", 0)
    diffs = comparison.get("differences", [])

    # Good layout match — stop
    if sim >= 70 and not diffs:
        return False

    # Layout-breaking issues — must fix
    layout_issues = [d for d in diffs if d.get("category") in ("wrong_layout", "missing_element", "wrong_proportions")]
    if layout_issues:
        logger.info(f"Copy revision needed: {len(layout_issues)} layout issues")
        return True

    # Low structural similarity — fix
    if sim < 60:
        logger.info(f"Copy revision needed: structural similarity {sim}% < 60%")
        return True

    # Acceptable structural match
    if sim >= 70:
        return False

    # Minor differences — fix if there are any
    if diffs:
        logger.info(f"Copy revision needed: {len(diffs)} differences, {sim}% similar")
        return True

    return False


def format_comparison_for_fix(comparison: dict) -> str:
    """Format comparison differences into fix instructions for Opus."""
    diffs = comparison.get("differences", [])
    if not diffs:
        return ""

    lines = ["COMPARISON TO REFERENCE — fix these differences:\n"]

    for d in diffs:
        category = d.get("category", "difference").upper().replace("_", " ")
        lines.append(
            f"[{category}] {d.get('what', '?')}\n"
            f"  Reference: {d.get('in_reference', '?')}\n"
            f"  Your render: {d.get('in_render', '?')}\n"
            f"  → Fix: {d.get('fix', 'Match the reference')}\n"
        )

    summary = comparison.get("summary", "")
    if summary:
        lines.append(f"\nPRIORITY: {summary}")

    return "\n".join(lines)
