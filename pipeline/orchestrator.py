"""
Pipeline orchestrator — runs all 8 steps in sequence.

Each step is independent. The orchestrator passes the accumulated
context forward and handles errors per step.
"""

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
from pipeline.steps.dynamic_template import generate_dynamic_template
from pipeline.steps.brain_write import brain_write

logger = logging.getLogger(__name__)


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

        # Step 6: Creative Decisions (Opus)
        await _notify("decisions", "Opus is picking headline, font, colors...")
        decisions = await creative_decisions(input, concept, copy, image, brain_ctx, previous_decisions)
        result.decisions = decisions
        await _notify("decisions", (
            f"Template: {decisions.template}, "
            f"Font: {decisions.font_headline} {decisions.font_headline_weight}"
        ))

        # Step 6b: Dynamic Template (Opus generates fresh layout from inspiration)
        await _notify("template", "Generating unique layout from your inspiration...")
        dynamic_html = await generate_dynamic_template(decisions, brain)
        await _notify("template", "Layout ready")

        # Step 7: Render (Playwright)
        await _notify("render", "Rendering final PNG...")
        render_result = await render(decisions, image, input.client, dynamic_html=dynamic_html)
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
