"""
Parse user feedback on taste analyses and store corrections/confirmations in Big Brain.

Handles:
- Confirmations: "yes", "exactly", "perfect" → store as confirmed
- Partial: "I like the font not the layout" → store correction
- Rejections: "no", "not this" → store in taste_rejected
- Client tagging: "this is for Somamed" → tag reference to client
"""

import json
import logging
import re

import anthropic

import config
from brain.client import Brain

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


FEEDBACK_SYSTEM_PROMPT = """You judge user feedback about a design analysis.

The user sent an inspiration image, the AI broke it down (typography, colors, composition, feeling), and now the user is responding. Your job is to understand their INTENT — they may write in English, Greek, or a mix, casually or formally.

CRITICAL DISTINCTION:
- Correcting a DETAIL within an aspect is NOT the same as rejecting that aspect.
- Example: "remove the electric lime" = the user likes the colors overall but wants one specific color removed. Colors should go in confirmed_aspects AND the correction should be noted.
- Example: "I hate the colors" = the user rejects the entire color palette. Colors should go in rejected_aspects.
- When in doubt, treat it as a correction (partial + confirmed) rather than a full rejection.

Possible intents:
1. CONFIRM — they agree with the analysis. Examples: "yes", "correct", "σωστά", "love it", "ακριβώς", "ok cool", "that's it", "nailed it", "👍", "ωραία", "τέλεια"
2. PARTIAL — they agree with some parts, want changes to others. This includes small corrections to a single detail. Examples: "colors are good but the font is wrong", "remove the lime green", "μ'αρέσουν τα χρώματα αλλά όχι η σύνθεση", "love the feeling, hate the layout", "make the font bolder"
3. REJECT — they disagree with EVERYTHING. Examples: "no", "not this", "delete it", "σβήσ'το", "nah", "completely wrong", "όχι καθόλου"
4. DIRECTION — they're giving new instructions. Examples: "make the font more editorial", "θέλω πιο ζεστά χρώματα", "text should be at the top"
5. CLIENT TAG — they're assigning this to a specific client. Examples: "this is for Somamed", "αυτό είναι για τον Μάκη", "use this for clinic clients"

A single message can combine multiple intents (e.g., "correct, this is for Somamed" = confirm + client tag).

Return ONLY valid JSON:
{
  "action": "confirm" | "partial" | "reject" | "direction",
  "confirmed_aspects": ["typography", "colors", "composition", "feeling"],
  "rejected_aspects": ["typography", "colors", "composition", "feeling"],
  "corrections": {"aspect": "what to change about it"},
  "client_tag": "client name or null",
  "summary": "one sentence summary of what the user meant"
}

When action is "confirm", confirmed_aspects should include ALL four aspects.
When action is "reject", rejected_aspects should include ALL four aspects.
For "partial":
- Aspects the user is HAPPY with (even if they want a small tweak) go in confirmed_aspects.
- Aspects the user EXPLICITLY dislikes entirely go in rejected_aspects.
- Aspects not mentioned at all go in confirmed_aspects (silence = approval).
- Any specific changes go in corrections with the aspect name as key.

No explanation outside JSON. No markdown wrapping."""


def _summarize_typography(typo) -> str:
    """Summarize typography for feedback prompt — handles both list and dict."""
    if isinstance(typo, list):
        cats = [t.get("font_category", "?") for t in typo]
        return ", ".join(set(cats)) if cats else "?"
    elif isinstance(typo, dict):
        return typo.get("font_category", "?")
    return "?"


async def parse_feedback(
    user_message: str,
    original_analysis: dict,
    brain: Brain,
    reference_id: str = None,
) -> dict:
    """
    Parse user feedback and store it in Big Brain.

    Args:
        user_message: The user's reply to the analysis.
        original_analysis: The full analysis dict that the user is responding to.
        brain: Brain client instance.
        reference_id: Optional brain entry ID of the original taste_reference.

    Returns:
        Parsed feedback dict with action and details.
    """
    # All feedback goes through Sonnet — no hardcoded word lists
    # Sonnet judges intent naturally in any language
    try:
        response = _client.messages.create(
            model=config.SONNET_MODEL,
            max_tokens=1024,
            system=FEEDBACK_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Original analysis summary:\n"
                        f"- Typography: {_summarize_typography(original_analysis.get('typography', []))}\n"
                        f"- Colors: {original_analysis.get('colors', {}).get('palette_mood', '?')}\n"
                        f"- Composition: {original_analysis.get('composition', {}).get('template_match', '?')}\n"
                        f"- Feeling: {original_analysis.get('feeling', {}).get('mood', '?')}\n"
                        f"- Layers: {len(original_analysis.get('layers', []))} layers\n\n"
                        f"User's feedback: \"{user_message}\""
                    ),
                }
            ],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        feedback = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to parse feedback with AI: {e}")
        feedback = {
            "action": "direction",
            "confirmed_aspects": [],
            "rejected_aspects": [],
            "corrections": {"general": user_message},
            "client_tag": None,
            "summary": user_message,
        }

    # Store in Big Brain based on action
    _store_feedback(brain, feedback, original_analysis, reference_id)

    return feedback


def _store_feedback(
    brain: Brain,
    feedback: dict,
    original_analysis: dict,
    reference_id: str = None,
) -> None:
    """Store parsed feedback in Big Brain."""
    action = feedback.get("action", "direction")
    client = feedback.get("client_tag") or "ALL"

    if action == "confirm":
        # Store confirmed aspects as high-weight taste data
        for aspect in feedback.get("confirmed_aspects", []):
            aspect_data = original_analysis.get(aspect, {})
            if aspect_data:
                brain.store(
                    topic=f"taste_{aspect}",
                    source="taste_engine",
                    content=json.dumps(aspect_data, ensure_ascii=False),
                    client=client,
                    summary=f"Confirmed {aspect} preference",
                    tags=["confirmed", aspect],
                )
        logger.info(f"Stored confirmed taste: {feedback.get('confirmed_aspects')}")

    elif action == "reject":
        brain.store(
            topic="taste_rejected",
            source="taste_engine",
            content=json.dumps(original_analysis, ensure_ascii=False),
            client=client,
            summary=feedback.get("summary", "User rejected this reference"),
            tags=["rejected"],
        )
        logger.info("Stored rejected taste reference")

    elif action in ("partial", "direction"):
        # Store corrections
        brain.store(
            topic="taste_correction",
            source="taste_engine",
            content=json.dumps({
                "feedback": feedback,
                "original_analysis": original_analysis,
            }, ensure_ascii=False),
            client=client,
            summary=feedback.get("summary", "User correction"),
            tags=["correction"] + feedback.get("confirmed_aspects", []),
        )

        # Also store confirmed aspects individually
        for aspect in feedback.get("confirmed_aspects", []):
            aspect_data = original_analysis.get(aspect, {})
            if aspect_data:
                brain.store(
                    topic=f"taste_{aspect}",
                    source="taste_engine",
                    content=json.dumps(aspect_data, ensure_ascii=False),
                    client=client,
                    summary=f"Confirmed {aspect} (partial)",
                    tags=["confirmed", aspect],
                )

        logger.info(f"Stored correction: {feedback.get('summary')}")

    # Tag to client if specified
    if feedback.get("client_tag") and reference_id:
        logger.info(f"Tagged reference {reference_id} to client: {feedback['client_tag']}")


def format_feedback_response(feedback: dict) -> str:
    """Format feedback acknowledgment for Telegram."""
    action = feedback.get("action", "direction")

    if action == "confirm":
        return "✅ Got it — saved these preferences as confirmed taste."
    elif action == "reject":
        return "❌ Understood — marked this as something you DON'T like. Won't repeat."
    elif action == "partial":
        confirmed = ", ".join(feedback.get("confirmed_aspects", []))
        rejected = ", ".join(feedback.get("rejected_aspects", []))
        corrections = feedback.get("corrections", {})
        msg = "📝 Noted —"
        if confirmed:
            msg += f" keeping {confirmed}."
        if rejected:
            msg += f" Dropping {rejected}."
        if corrections:
            msg += "\nChanges noted:"
            for aspect, correction in corrections.items():
                msg += f"\n  • {aspect}: {correction}"
        return msg
    else:
        return f"📝 Stored your direction: {feedback.get('summary', 'noted')}"
