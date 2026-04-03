"""Curated font pools with category rotation for creative variety.

All fonts verified on Google Fonts with correct weight ranges and language support.
Categories rotate to prevent Opus from defaulting to the same 3-4 fonts.
"""

# ── Greek-supporting fonts (verified on Google Fonts) ──────────────

GREEK_FONTS = {
    "geometric_sans": [
        {"name": "Inter", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Outfit", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Urbanist", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Albert Sans", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Rubik", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Nunito Sans", "weights": [300, 400, 600, 700, 800, 900]},
        {"name": "Lexend", "weights": [300, 400, 500, 600, 700, 800]},
    ],
    "humanist_sans": [
        {"name": "DM Sans", "weights": [400, 500, 700]},
        {"name": "Open Sans", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Manrope", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Source Sans 3", "weights": [300, 400, 600, 700, 900]},
        {"name": "Karla", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Work Sans", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Cabin", "weights": [400, 500, 600, 700]},
        {"name": "Lato", "weights": [300, 400, 700, 900]},
    ],
    "neo_grotesque": [
        {"name": "Roboto", "weights": [300, 400, 500, 700, 900]},
        {"name": "IBM Plex Sans", "weights": [300, 400, 500, 600, 700]},
        {"name": "Noto Sans", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Barlow", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Barlow Condensed", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Roboto Condensed", "weights": [300, 400, 700]},
    ],
    "serif": [
        {"name": "Noto Serif", "weights": [400, 700]},
        {"name": "EB Garamond", "weights": [400, 500, 600, 700, 800]},
        {"name": "Literata", "weights": [400, 500, 600, 700, 800, 900]},
        {"name": "Cardo", "weights": [400, 700]},
        {"name": "GFS Didot", "weights": [400]},
        {"name": "Cormorant", "weights": [300, 400, 500, 600, 700]},
    ],
    "display": [
        {"name": "Epilogue", "weights": [400, 500, 600, 700, 800, 900]},
        {"name": "Syne", "weights": [400, 500, 600, 700, 800]},
        {"name": "Commissioner", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Josefin Sans", "weights": [300, 400, 600, 700]},
        {"name": "Exo 2", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Jost", "weights": [300, 400, 500, 600, 700, 800, 900]},
    ],
    "condensed": [
        {"name": "Barlow Condensed", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Roboto Condensed", "weights": [300, 400, 700]},
        {"name": "Saira Condensed", "weights": [300, 400, 500, 600, 700, 800, 900]},
    ],
}

# ── Latin-only fonts (broader pool, no Greek needed) ──────────────

LATIN_FONTS = {
    "geometric_sans": [
        {"name": "Inter", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Outfit", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Urbanist", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Plus Jakarta Sans", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Figtree", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "General Sans", "weights": [400, 500, 600, 700]},
        {"name": "Satoshi", "weights": [400, 500, 700, 900]},
        {"name": "Space Grotesk", "weights": [300, 400, 500, 600, 700]},
    ],
    "humanist_sans": [
        {"name": "DM Sans", "weights": [400, 500, 700]},
        {"name": "Manrope", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Karla", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Work Sans", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Nunito", "weights": [300, 400, 600, 700, 800, 900]},
    ],
    "neo_grotesque": [
        {"name": "IBM Plex Sans", "weights": [300, 400, 500, 600, 700]},
        {"name": "Barlow", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Archivo", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Overpass", "weights": [300, 400, 600, 700, 800, 900]},
    ],
    "serif": [
        {"name": "Playfair Display", "weights": [400, 500, 600, 700, 800, 900]},
        {"name": "DM Serif Display", "weights": [400]},
        {"name": "Cormorant", "weights": [300, 400, 500, 600, 700]},
        {"name": "Fraunces", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Spectral", "weights": [300, 400, 500, 600, 700, 800]},
        {"name": "Instrument Serif", "weights": [400]},
        {"name": "Lora", "weights": [400, 500, 600, 700]},
        {"name": "Bitter", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Libre Baskerville", "weights": [400, 700]},
    ],
    "display": [
        {"name": "Syne", "weights": [400, 500, 600, 700, 800]},
        {"name": "Epilogue", "weights": [400, 500, 600, 700, 800, 900]},
        {"name": "Clash Display", "weights": [400, 500, 600, 700]},
        {"name": "Cabinet Grotesk", "weights": [400, 500, 700, 800, 900]},
        {"name": "Unbounded", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Josefin Sans", "weights": [300, 400, 600, 700]},
        {"name": "Bebas Neue", "weights": [400]},
        {"name": "Oswald", "weights": [300, 400, 500, 600, 700]},
        {"name": "Anton", "weights": [400]},
    ],
    "condensed": [
        {"name": "Barlow Condensed", "weights": [300, 400, 500, 600, 700, 800, 900]},
        {"name": "Roboto Condensed", "weights": [300, 400, 700]},
        {"name": "Oswald", "weights": [300, 400, 500, 600, 700]},
        {"name": "Bebas Neue", "weights": [400]},
        {"name": "Saira Condensed", "weights": [300, 400, 500, 600, 700, 800, 900]},
    ],
    "handwritten_script": [
        {"name": "Caveat", "weights": [400, 500, 600, 700]},
        {"name": "Kalam", "weights": [300, 400, 700]},
        {"name": "Permanent Marker", "weights": [400]},
        {"name": "Patrick Hand", "weights": [400]},
    ],
}

# All category names
FONT_CATEGORIES = ["geometric_sans", "humanist_sans", "neo_grotesque", "serif", "display", "condensed"]
LATIN_EXTRA_CATEGORIES = FONT_CATEGORIES + ["handwritten_script"]


def _detect_greek(text: str) -> bool:
    """Check if text contains Greek characters."""
    return any('\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF' for c in text)


def build_font_instruction(
    headlines: list[str],
    previous_fonts: list[str] | None = None,
    scout_fonts: list[str] | None = None,
) -> str:
    """Build a font instruction block for the decisions prompt.

    Args:
        headlines: The headline options (to detect Greek)
        previous_fonts: Fonts used in recent posts (to ban)
        scout_fonts: Trending fonts discovered by design scout
    """
    # Detect if we need Greek support
    combined_text = " ".join(headlines)
    needs_greek = _detect_greek(combined_text)

    pool = GREEK_FONTS if needs_greek else LATIN_FONTS
    categories = FONT_CATEGORIES if needs_greek else LATIN_EXTRA_CATEGORIES

    # Determine which category to push (rotate away from recent)
    banned_categories = set()
    banned_fonts = set()
    if previous_fonts:
        banned_fonts = set(previous_fonts[-3:])  # Ban last 3 fonts
        # Figure out which categories the recent fonts belong to
        for font_name in previous_fonts[-2:]:  # Check last 2 for category
            for cat, fonts in pool.items():
                for f in fonts:
                    if f["name"] == font_name:
                        banned_categories.add(cat)

    # Pick suggested categories (ones not recently used)
    available_categories = [c for c in categories if c not in banned_categories and c in pool]
    if not available_categories:
        available_categories = categories  # Reset if all banned

    # Build the instruction
    lines = []

    if needs_greek:
        lines.append("⚠️ TEXT CONTAINS GREEK — you MUST use fonts with Greek subset support.")
        lines.append("AVAILABLE FONTS BY CATEGORY (pick from these):\n")
    else:
        lines.append("AVAILABLE FONTS BY CATEGORY (pick from these, or any Google Font):\n")

    for cat in categories:
        if cat not in pool:
            continue
        fonts = pool[cat]
        cat_label = cat.replace("_", " ").title()
        is_suggested = cat in available_categories and cat not in banned_categories
        marker = " ⭐ SUGGESTED" if is_suggested else ""
        font_names = [f["name"] for f in fonts]
        lines.append(f"  {cat_label}{marker}: {', '.join(font_names)}")

    # Add scout-discovered fonts if any
    if scout_fonts:
        lines.append(f"\n  🔍 Trending (from design scout): {', '.join(scout_fonts)}")

    # Variety enforcement
    lines.append("")
    if banned_fonts:
        banned_str = ", ".join(banned_fonts)
        lines.append(f"🚫 DO NOT use these fonts (recently used): {banned_str}")

    if banned_categories:
        banned_cat_str = ", ".join(c.replace("_", " ").title() for c in banned_categories)
        suggested_cat_str = ", ".join(c.replace("_", " ").title() for c in available_categories[:2])
        lines.append(f"🚫 AVOID these categories (recently used): {banned_cat_str}")
        lines.append(f"✅ TRY these categories instead: {suggested_cat_str}")

    lines.append("\nBe creative with your font choice. Don't default to Inter or DM Sans.")
    lines.append("Mix serif headlines with sans-serif body, or try display/condensed for impact.")

    return "\n".join(lines)


def validate_font_weight(font_name: str, requested_weight: int, needs_greek: bool = False) -> int:
    """Validate that a font supports the requested weight. Returns closest valid weight."""
    pool = GREEK_FONTS if needs_greek else LATIN_FONTS

    for cat, fonts in pool.items():
        for f in fonts:
            if f["name"] == font_name:
                if requested_weight in f["weights"]:
                    return requested_weight
                # Find closest weight
                return min(f["weights"], key=lambda w: abs(w - requested_weight))

    # Font not in our pool — assume any weight works (Google Fonts has variable fonts)
    return requested_weight


def get_scout_font_names(brain) -> list[str]:
    """Extract font names from design scout entries in Brain."""
    try:
        entries = brain.query(topic="design_scout", limit=3)
        if not entries:
            return []

        # Look for font names in scout content
        import re
        font_names = set()
        for entry in entries:
            content = entry.get("content", "")
            # Match patterns like "Font Name" after Typography: or font:
            matches = re.findall(
                r'(?:typography|font|typeface)[:\s]+\*?\*?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
                content, re.IGNORECASE
            )
            for m in matches:
                name = m.strip()
                # Filter out common false positives
                if name not in ("The", "For", "This", "Best", "New", "Use", "Bold", "Light", "Regular", "Medium"):
                    font_names.add(name)

        return list(font_names)[:8]  # Cap at 8
    except Exception:
        return []
