"""
Pipeline orchestrator — runs all 8 steps in sequence.

Each step is independent. The orchestrator passes the accumulated
context forward and handles errors per step.
"""

import json
import logging
import time

from brain.client import Brain
from pipeline.types import PipelineInput, PipelineResult
from pipeline.steps.research import research
from pipeline.steps.brain_read import brain_read
from pipeline.steps.concept import creative_concept
from pipeline.steps.copy import write_copy
from pipeline.steps.image_gen import generate_image
from pipeline.steps.decisions import creative_decisions
from pipeline.steps.render import render
from pipeline.steps.dynamic_template import generate_dynamic_template, get_client_preferences
from pipeline.steps.brain_write import brain_write

logger = logging.getLogger(__name__)


def _get_client_logo(brain: Brain, client: str) -> str | None:
    """Fetch the latest logo for a client from the Brain. Returns base64 string or None."""
    logos = brain.query(topic="brand_logo", client=client, limit=1)
    if not logos:
        # Also try with "ALL" as fallback
        return None
    try:
        data = json.loads(logos[0]["content"])
        return data.get("image_b64")
    except (json.JSONDecodeError, KeyError):
        return None


async def run_pipeline(
    input: PipelineInput,
    brain: Brain,
    on_progress=None,
    previous_decisions: list[dict] | None = None,
) -> PipelineResult:
    """
    Run the full creative generation pipeline.

    Args:
        input: PipelineInput with client, brief, platform.
        brain: Brain client instance.
        on_progress: Optional async callback(step_name, message) for live updates.

    Returns:
        PipelineResult with success flag, image path, and all decisions.
    """
    start = time.time()
    result = PipelineResult()

    async def _notify(step: str, msg: str):
        logger.info(f"[{step}] {msg}")
        if on_progress:
            try:
                await on_progress(step, msg)
            except Exception:
                pass

    try:
        # Step 1: Research
        await _notify("research", "Searching for trends...")
        research_result = await research(input)
        await _notify("research", f"Found trends ({len(research_result.trends)} chars)")

        # Step 2: Brain Read
        await _notify("brain", "Reading client profile + taste data...")
        brain_ctx = await brain_read(input, brain)
        await _notify("brain", (
            f"Client: {'✓' if brain_ctx.client_profile else '✗'}, "
            f"Past concepts: {len(brain_ctx.past_concepts)}, "
            f"Taste refs: {brain_ctx.taste_context.get('total_references', 0)}"
        ))

        # Step 3: Creative Concept (Opus)
        await _notify("concept", "Opus is thinking about the visual concept...")
        concept = await creative_concept(input, research_result, brain_ctx)
        result.concept = concept
        await _notify("concept", f"Concept: {concept.object}")

        # Step 4: Copy (Sonnet)
        await _notify("copy", "Sonnet is writing headlines...")
        copy = await write_copy(input, concept, brain_ctx)
        result.copy = copy
        await _notify("copy", f"{len(copy.headlines)} headlines ready")

        # Step 5: Image Generation (Opus prompt → Flux/Ideogram)
        await _notify("image", "Generating hero image...")
        image = await generate_image(concept, brain_ctx)
        await _notify("image", f"Image ready ({image.model_used})")

        result.hero_image = image

        # Fetch client-specific preferences (liked colors, fonts, rules)
        client_prefs = get_client_preferences(brain, input.client)
        if client_prefs:
            await _notify("decisions", f"Found preferences for {input.client}")

        # Step 6: Creative Decisions (Opus)
        await _notify("decisions", "Opus is picking headline, font, colors...")
        decisions = await creative_decisions(input, concept, copy, image, brain_ctx, previous_decisions, client_prefs)
        result.decisions = decisions
        await _notify("decisions", (
            f"Template: {decisions.template}, "
            f"Font: {decisions.font_headline} {decisions.font_headline_weight}"
        ))

        # Fetch client logo if available
        logo_b64 = _get_client_logo(brain, input.client)
        has_logo = logo_b64 is not None
        if has_logo:
            await _notify("template", f"Found logo for {input.client}")

        # Step 6b: Dynamic Template (Opus generates fresh layout from inspiration)
        await _notify("template", "Generating unique layout from your inspiration...")
        dynamic_html = await generate_dynamic_template(decisions, brain, has_logo=has_logo)
        await _notify("template", "Layout ready")

        # Step 7: Render (Playwright)
        await _notify("render", "Rendering final PNG...")
        render_result = await render(decisions, image, input.client, dynamic_html=dynamic_html, logo_b64=logo_b64)
        result.image_path = render_result.final_image_path
        await _notify("render", f"Rendered: {render_result.final_image_path}")

        # Step 8: Brain Write
        await _notify("brain_write", "Storing concept in Big Brain...")
        await brain_write(input, concept, copy, decisions, image, brain)
        await _notify("brain_write", "Stored ✓")

        result.success = True
        elapsed = time.time() - start
        logger.info(f"Pipeline complete in {elapsed:.1f}s: {result.image_path}")

    except Exception as e:
        result.success = False
        result.error = str(e)
        elapsed = time.time() - start
        logger.error(f"Pipeline failed after {elapsed:.1f}s: {e}")

    return result


async def run_carousel(
    input: PipelineInput,
    brain: Brain,
    count: int = 6,
    on_progress=None,
) -> list[PipelineResult]:
    """
    Generate a carousel of cohesive posts.

    The first post is generated normally. Posts 2-N reuse the same
    design decisions (font, colors, template HTML) but get fresh
    concepts, headlines, and images.
    """
    start = time.time()

    async def _notify(step: str, msg: str):
        logger.info(f"[carousel] [{step}] {msg}")
        if on_progress:
            try:
                await on_progress(step, msg)
            except Exception:
                pass

    results = []

    # ── Shared setup (done once) ──────────────────────────
    await _notify("research", "Researching trends...")
    research_result = await research(input)

    await _notify("brain", "Reading client profile + taste data...")
    brain_ctx = await brain_read(input, brain)

    client_prefs = get_client_preferences(brain, input.client)

    # Fetch client logo
    logo_b64 = _get_client_logo(brain, input.client)

    # ── First post: full pipeline to establish the design ──
    await _notify("carousel", f"Generating post 1/{count}...")
    first = await _generate_carousel_slide(
        input, research_result, brain_ctx, client_prefs, brain,
        slide_num=1, total=count,
        locked_decisions=None, locked_template_html=None,
        on_progress=on_progress, logo_b64=logo_b64,
    )
    results.append(first)

    if not first.success:
        return results

    # Lock the design from post 1
    locked_decisions = first.decisions
    # Generate the template HTML once and reuse
    locked_html = await generate_dynamic_template(locked_decisions, brain, has_logo=logo_b64 is not None)

    # ── Posts 2-N: new concept + image, same design ──
    for i in range(2, count + 1):
        await _notify("carousel", f"Generating post {i}/{count}...")

        # Modify brief to request a different concept
        varied_input = PipelineInput(
            client=input.client,
            brief=f"{input.brief} — THIS IS POST {i} OF {count} IN A CAROUSEL. "
                  f"Use a DIFFERENT visual concept from the previous slides. "
                  f"Previous concepts: {', '.join(r.concept.object for r in results if r.concept)}",
            platform=input.platform,
        )

        slide = await _generate_carousel_slide(
            varied_input, research_result, brain_ctx, client_prefs, brain,
            slide_num=i, total=count,
            locked_decisions=locked_decisions, locked_template_html=locked_html,
            on_progress=on_progress, logo_b64=logo_b64,
        )
        results.append(slide)

    elapsed = time.time() - start
    success_count = sum(1 for r in results if r.success)
    logger.info(f"Carousel complete: {success_count}/{count} posts in {elapsed:.1f}s")

    return results


async def _generate_carousel_slide(
    input: PipelineInput,
    research_result,
    brain_ctx,
    client_prefs: dict | None,
    brain: Brain,
    slide_num: int,
    total: int,
    locked_decisions=None,
    locked_template_html: str | None = None,
    on_progress=None,
    logo_b64: str | None = None,
) -> PipelineResult:
    """Generate a single carousel slide."""
    result = PipelineResult()

    async def _notify(step: str, msg: str):
        logger.info(f"[slide {slide_num}/{total}] [{step}] {msg}")
        if on_progress:
            try:
                await on_progress(step, msg)
            except Exception:
                pass

    try:
        # Always generate fresh concept
        concept = await creative_concept(input, research_result, brain_ctx)
        result.concept = concept
        await _notify("concept", f"Concept: {concept.object[:50]}")

        # Always generate fresh copy
        copy = await write_copy(input, concept, brain_ctx)
        result.copy = copy

        # Always generate fresh image
        image = await generate_image(concept, brain_ctx)
        result.hero_image = image
        await _notify("image", f"Image ready ({image.model_used})")

        if locked_decisions:
            # Reuse design but pick best headline and new subtext/CTA
            decisions = await creative_decisions(
                input, concept, copy, image, brain_ctx,
                client_preferences=client_prefs,
            )
            # Lock the design choices from slide 1
            decisions.font_headline = locked_decisions.font_headline
            decisions.font_headline_weight = locked_decisions.font_headline_weight
            decisions.font_headline_size = locked_decisions.font_headline_size
            decisions.font_headline_tracking = locked_decisions.font_headline_tracking
            decisions.font_headline_line_height = locked_decisions.font_headline_line_height
            decisions.font_headline_case = locked_decisions.font_headline_case
            decisions.font_subtext = locked_decisions.font_subtext
            decisions.font_subtext_weight = locked_decisions.font_subtext_weight
            decisions.font_subtext_size = locked_decisions.font_subtext_size
            decisions.color_bg = locked_decisions.color_bg
            decisions.color_text = locked_decisions.color_text
            decisions.color_accent = locked_decisions.color_accent
            decisions.color_subtext = locked_decisions.color_subtext
            decisions.template = locked_decisions.template
        else:
            # First slide: full decisions
            decisions = await creative_decisions(
                input, concept, copy, image, brain_ctx,
                client_preferences=client_prefs,
            )

        result.decisions = decisions

        # Render
        if locked_template_html:
            render_result = await render(decisions, image, input.client, dynamic_html=locked_template_html, logo_b64=logo_b64)
        else:
            dynamic_html = await generate_dynamic_template(decisions, brain, has_logo=logo_b64 is not None)
            render_result = await render(decisions, image, input.client, dynamic_html=dynamic_html, logo_b64=logo_b64)

        result.image_path = render_result.final_image_path
        await _notify("render", f"Rendered slide {slide_num}")

        # Store in brain
        await brain_write(input, concept, copy, decisions, image, brain)

        result.success = True

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error(f"Slide {slide_num} failed: {e}")

    return result


def format_result_for_telegram(result: PipelineResult, input: PipelineInput) -> str:
    """Format pipeline result for Telegram message."""
    if not result.success:
        return f"❌ Generation failed: {result.error[:300]}"

    lines = [f"✅ <b>Post generated for {input.client}</b>\n"]

    if result.concept:
        lines.append(f"🎨 <b>Concept:</b> {result.concept.object}")
        lines.append(f"   <i>{result.concept.why}</i>\n")

    if result.decisions:
        d = result.decisions
        lines.append(f"📝 <b>Headline:</b> {d.headline}")
        if d.headline_reason:
            lines.append(f"   <i>Why: {d.headline_reason}</i>")
        lines.append(f"\n🔤 <b>Font:</b> {d.font_headline} ({d.font_headline_weight})")
        lines.append(f"🎨 <b>Colors:</b> bg {d.color_bg}, text {d.color_text}, accent {d.color_accent}")
        lines.append(f"📐 <b>Template:</b> {d.template}")

    if result.copy and result.decisions:
        if result.decisions.subtext:
            lines.append(f"\n💬 <b>Subtext:</b> {result.decisions.subtext}")
        if result.decisions.cta:
            lines.append(f"👆 <b>CTA:</b> {result.decisions.cta}")

    return "\n".join(lines)
