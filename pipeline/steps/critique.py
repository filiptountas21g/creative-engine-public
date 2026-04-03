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

    # Build taste context — reviewer checks if design matches current aesthetic
    taste_section = ""
    if taste_context:
        taste_section = (
            f"\n\nDESIGNER'S CURRENT TASTE PROFILE (recent preferences weighted higher):\n"
            f"{taste_context}\n"
            f"Consider whether this design feels aligned with this aesthetic. "
            f"If it's drifting away from these preferences without a good reason, flag it."
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
