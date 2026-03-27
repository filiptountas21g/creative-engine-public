"""
Taste memory — higher-level queries over accumulated taste data.

Provides:
- get_taste_context() → consolidated taste profile for creative pipeline
- get_taste_summary() → human-readable summary for Telegram /taste command
- detect_patterns() → find recurring preferences across references
"""

import json
import logging
from collections import Counter

from brain.client import Brain

logger = logging.getLogger(__name__)


async def get_taste_context(brain: Brain, client: str = None) -> dict:
    """
    Returns consolidated taste profile for injection into creative prompts.
    Used by the generation pipeline (Step 2) to inform Opus.

    Args:
        brain: Brain client instance.
        client: Optional client name for client-specific taste.

    Returns:
        Dict with preferred_fonts, preferred_colors, preferred_composition,
        preferred_feeling, confirmed_rules, avoid, client_specific.
    """
    # Fetch all taste data
    all_refs = brain.query(topic="taste_reference", limit=50)
    confirmed_typo = brain.query(topic="taste_typography", limit=30)
    confirmed_colors = brain.query(topic="taste_colors", limit=30)
    confirmed_comp = brain.query(topic="taste_composition", limit=30)
    corrections = brain.query(topic="taste_correction", limit=20)
    rejected = brain.query(topic="taste_rejected", limit=20)

    # Extract patterns
    font_categories = []
    color_temps = []
    compositions = []
    moods = []
    rules = []

    for ref in all_refs:
        try:
            data = json.loads(ref["content"])
            typo = data.get("typography", {})
            colors = data.get("colors", {})
            comp = data.get("composition", {})
            feel = data.get("feeling", {})

            if typo.get("font_category"):
                font_categories.append(typo["font_category"])
            if colors.get("temperature"):
                color_temps.append(colors["temperature"])
            if comp.get("template_match"):
                compositions.append(comp["template_match"])
            if feel.get("mood"):
                moods.append(feel["mood"])
            for rule in data.get("reusable_rules", []):
                rules.append(rule)
        except (json.JSONDecodeError, KeyError):
            continue

    # Add confirmed data (higher weight)
    for entry in confirmed_typo:
        try:
            data = json.loads(entry["content"])
            if data.get("font_category"):
                font_categories.extend([data["font_category"]] * 2)  # double weight for confirmed
        except (json.JSONDecodeError, KeyError):
            continue

    # Build avoid list from rejections
    avoid = []
    for entry in rejected:
        try:
            data = json.loads(entry["content"])
            feel = data.get("feeling", {})
            if feel.get("mood"):
                avoid.append(feel["mood"])
            comp = data.get("composition", {})
            if comp.get("template_match"):
                avoid.append(f"{comp['template_match']} layout")
        except (json.JSONDecodeError, KeyError):
            continue

    # Build corrections summary
    correction_notes = []
    for entry in corrections:
        try:
            data = json.loads(entry["content"])
            feedback = data.get("feedback", {})
            if feedback.get("summary"):
                correction_notes.append(feedback["summary"])
        except (json.JSONDecodeError, KeyError):
            continue

    # Client-specific preferences
    client_specific = {}
    if client:
        client_refs = brain.query(topic="taste_reference", client=client, limit=20)
        client_corrections = brain.query(topic="taste_correction", client=client, limit=10)
        if client_refs or client_corrections:
            client_specific[client] = _summarize_client_taste(client_refs, client_corrections)

    # Count most common
    top_fonts = [f for f, _ in Counter(font_categories).most_common(3)]
    top_temps = [t for t, _ in Counter(color_temps).most_common(2)]
    top_comps = [c for c, _ in Counter(compositions).most_common(3)]
    top_moods = [m for m, _ in Counter(moods).most_common(3)]
    top_rules = [r for r, _ in Counter(rules).most_common(5)]

    return {
        "preferred_fonts": top_fonts,
        "preferred_color_temperatures": top_temps,
        "preferred_compositions": top_comps,
        "preferred_moods": top_moods,
        "confirmed_rules": top_rules,
        "avoid": list(set(avoid)),
        "corrections": correction_notes[-5:],  # last 5 corrections
        "client_specific": client_specific,
        "total_references": len(all_refs),
    }


def _summarize_client_taste(refs: list, corrections: list) -> str:
    """Summarize taste for a specific client."""
    moods = []
    for ref in refs:
        try:
            data = json.loads(ref["content"])
            mood = data.get("feeling", {}).get("mood", "")
            if mood:
                moods.append(mood)
        except (json.JSONDecodeError, KeyError):
            continue

    correction_notes = []
    for entry in corrections:
        try:
            data = json.loads(entry["content"])
            summary = data.get("feedback", {}).get("summary", "")
            if summary:
                correction_notes.append(summary)
        except (json.JSONDecodeError, KeyError):
            continue

    parts = []
    if moods:
        parts.append(f"Moods: {', '.join(set(moods))}")
    if correction_notes:
        parts.append(f"Corrections: {'; '.join(correction_notes[-3:])}")

    return ". ".join(parts) if parts else "No specific preferences yet."


async def get_taste_summary(brain: Brain, aspect: str = None) -> str:
    """
    Human-readable taste summary for Telegram /taste command.

    Args:
        brain: Brain client.
        aspect: Optional filter — "fonts", "colors", "composition", "feeling".

    Returns:
        Formatted string for Telegram.
    """
    ctx = await get_taste_context(brain)

    if aspect == "fonts":
        fonts = ctx.get("preferred_fonts", [])
        if not fonts:
            return "📝 No font preferences stored yet. Send some inspiration images!"
        return (
            "📝 <b>Your Typography Preferences</b>\n\n"
            + "\n".join(f"  • {f}" for f in fonts)
            + f"\n\n<i>Based on {ctx['total_references']} analyzed references.</i>"
        )

    if aspect == "colors":
        temps = ctx.get("preferred_color_temperatures", [])
        if not temps:
            return "🎨 No color preferences stored yet. Send some inspiration images!"
        return (
            "🎨 <b>Your Color Preferences</b>\n\n"
            f"  • Temperature: {', '.join(temps)}\n"
            f"\n<i>Based on {ctx['total_references']} analyzed references.</i>"
        )

    # Full summary
    if ctx["total_references"] == 0:
        return (
            "🧠 <b>Taste Engine</b>\n\n"
            "No references analyzed yet.\n"
            "Send me inspiration images or drop them in the Drive folder!"
        )

    lines = [f"🧠 <b>Your Taste Profile</b> ({ctx['total_references']} references)\n"]

    if ctx.get("preferred_fonts"):
        lines.append("📝 <b>Fonts:</b> " + ", ".join(ctx["preferred_fonts"]))
    if ctx.get("preferred_color_temperatures"):
        lines.append("🎨 <b>Colors:</b> " + ", ".join(ctx["preferred_color_temperatures"]))
    if ctx.get("preferred_compositions"):
        lines.append("📐 <b>Layouts:</b> " + ", ".join(ctx["preferred_compositions"]))
    if ctx.get("preferred_moods"):
        lines.append("✨ <b>Moods:</b> " + ", ".join(ctx["preferred_moods"]))

    if ctx.get("confirmed_rules"):
        lines.append("\n📋 <b>Your Rules:</b>")
        for rule in ctx["confirmed_rules"]:
            lines.append(f"  • {rule}")

    if ctx.get("avoid"):
        lines.append("\n🚫 <b>Avoid:</b> " + ", ".join(ctx["avoid"]))

    return "\n".join(lines)
