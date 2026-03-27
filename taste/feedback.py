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


FEEDBACK_SYSTEM_PROMPT = """You parse user feedback about a design analysis.

The user is responding to a breakdown of an inspiration image. They may:
1. Confirm the whole analysis: "yes", "exactly", "perfect", "love it"
2. Partially agree: "I like the font but not the layout", "colors are good, composition is wrong"
3. Reject everything: "no", "not this", "I don't like it"
4. Tag to a client: "this is for Somamed", "use this for clinic clients"
5. Give specific corrections: "the font should be more editorial", "I prefer text at the top"

Return JSON with:
{
  "action": "confirm" | "partial" | "reject" | "direction",
  "confirmed_aspects": ["typography", "colors", "composition", "feeling"],  // which aspects the user confirmed
  "rejected_aspects": ["typography", "colors", "composition", "feeling"],  // which aspects the user rejected
  "corrections": {"aspect": "correction text"},  // specific corrections
  "client_tag": "client name or null",  // if user tagged a client
  "summary": "one sentence summary of what the user said"
}

Only return valid JSON. No explanation outside JSON."""


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
    # Quick pattern matching for simple responses (save an API call)
    lower = user_message.strip().lower()

    if lower in ("yes", "exactly", "perfect", "love it", "yes exactly", "ναι", "τέλειο"):
        feedback = {
            "action": "confirm",
            "confirmed_aspects": ["typography", "colors", "composition", "feeling"],
            "rejected_aspects": [],
            "corrections": {},
            "client_tag": None,
            "summary": "User confirmed the full analysis.",
        }
    elif lower in ("no", "not this", "no way", "delete", "όχι"):
        feedback = {
            "action": "reject",
            "confirmed_aspects": [],
            "rejected_aspects": ["typography", "colors", "composition", "feeling"],
            "corrections": {},
            "client_tag": None,
            "summary": "User rejected the full analysis.",
        }
    else:
        # Use Sonnet to parse complex feedback (cheaper than Opus)
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
                            f"- Typography: {original_analysis.get('typography', {}).get('font_category', '?')}\n"
                            f"- Colors: {original_analysis.get('colors', {}).get('palette_mood', '?')}\n"
                            f"- Composition: {original_analysis.get('composition', {}).get('template_match', '?')}\n"
                            f"- Feeling: {original_analysis.get('feeling', {}).get('mood', '?')}\n\n"
                            f"User's feedback: \"{user_message}\""
                        ),
                    }
                ],
            )
            raw = response.content[0].text.strip()
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
        msg = "📝 Noted —"
        if confirmed:
            msg += f" keeping {confirmed}."
        if rejected:
            msg += f" Dropping {rejected}."
        corrections = feedback.get("corrections", {})
        if corrections:
            msg += "\nCorrections stored:"
            for aspect, correction in corrections.items():
                msg += f"\n  • {aspect}: {correction}"
        return msg
    else:
        return f"📝 Stored your direction: {feedback.get('summary', 'noted')}"
