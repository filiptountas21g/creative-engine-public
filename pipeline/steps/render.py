"""Step 7: Render — Playwright loads HTML template, injects decisions, screenshots to PNG."""

import logging
import time
from pathlib import Path
from urllib.parse import quote

from pipeline.types import CreativeDecisions, ImageResult, RenderResult
import config

logger = logging.getLogger(__name__)


def _build_font_url(font_name: str, weight: int) -> str:
    """Build a Google Fonts CSS2 URL."""
    encoded = quote(font_name)
    return f"https://fonts.googleapis.com/css2?family={encoded}:wght@{weight}&display=swap"


def _inject_into_template(
    template_html: str,
    decisions: CreativeDecisions,
    image: ImageResult,
    client_name: str,
    logo_b64: str | None = None,
) -> str:
    """Replace placeholders and CSS variables in the HTML template."""
    # Build Google Fonts URL (both headline and subtext)
    font_urls = []
    font_urls.append(_build_font_url(decisions.font_headline, decisions.font_headline_weight))
    if decisions.font_subtext != decisions.font_headline:
        font_urls.append(_build_font_url(decisions.font_subtext, decisions.font_subtext_weight))

    font_links = "\n".join(
        f'<link href="{url}" rel="stylesheet">' for url in font_urls
    )

    # Replace placeholders
    html = template_html
    html = html.replace("{{FONT_URL}}", font_urls[0])  # primary font
    html = html.replace("{{HEADLINE}}", decisions.headline)
    html = html.replace("{{SUBTEXT}}", decisions.subtext)
    html = html.replace("{{CTA}}", decisions.cta)
    html = html.replace("{{CLIENT_NAME}}", client_name.upper())

    # Image — embed as base64 data URI so it always loads in Playwright
    img_path = Path(image.image_path).resolve()
    import base64
    try:
        img_bytes = img_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_src = f"data:image/png;base64,{img_b64}"
    except Exception:
        img_src = f"file://{img_path}"
    html = html.replace("{{IMAGE_PATH}}", img_src)

    # Logo — inject if available, otherwise remove placeholder
    if logo_b64 and "{{LOGO_PATH}}" in html:
        logo_src = f"data:image/png;base64,{logo_b64}"
        html = html.replace("{{LOGO_PATH}}", logo_src)
    elif "{{LOGO_PATH}}" in html:
        # Remove the logo img tag entirely if no logo
        import re
        html = re.sub(r'<img[^>]*\{\{LOGO_PATH\}\}[^>]*/?\s*>', '', html)

    # Inject CSS variable overrides
    css_overrides = f"""
    <style>
      :root {{
        --font-headline: '{decisions.font_headline}';
        --font-headline-weight: {decisions.font_headline_weight};
        --font-headline-size: {decisions.font_headline_size}px;
        --font-headline-tracking: {decisions.font_headline_tracking};
        --font-headline-line-height: {decisions.font_headline_line_height};
        --font-headline-case: {decisions.font_headline_case};
        --font-subtext: '{decisions.font_subtext}';
        --font-subtext-weight: {decisions.font_subtext_weight};
        --font-subtext-size: {decisions.font_subtext_size}px;
        --color-bg: {decisions.color_bg};
        --color-text: {decisions.color_text};
        --color-accent: {decisions.color_accent};
        --color-subtext: {decisions.color_subtext};
        --headline-margin-x: {decisions.headline_margin_x}px;
        --headline-margin-y: {decisions.headline_margin_y}px;
        --headline-max-width: {decisions.headline_max_width};
        --image-padding: {decisions.image_padding}px;
        --image-object-fit: contain;
        --client-color: {decisions.color_accent};
      }}
    </style>
    """

    # Inject font links and CSS overrides into <head>
    if "<head>" in html:
        html = html.replace("<head>", f"<head>\n{font_links}\n{css_overrides}")
    else:
        html = f"<head>\n{font_links}\n{css_overrides}\n</head>\n{html}"

    return html


async def render(
    decisions: CreativeDecisions,
    image: ImageResult,
    client_name: str,
    dynamic_html: str | None = None,
    logo_b64: str | None = None,
) -> RenderResult:
    """
    Render the final post as a 1080x1080 PNG.

    If dynamic_html is provided, uses that directly.
    Otherwise loads from the template file on disk.
    """
    from playwright.async_api import async_playwright

    # Use dynamic HTML if provided, otherwise load from file
    if dynamic_html:
        template_html = dynamic_html
        logger.info("Using dynamic template")
    else:
        template_path = Path(config.TEMPLATES_DIR) / f"{decisions.template}.html"
        if not template_path.exists():
            logger.warning(f"Template {decisions.template} not found — using fallback")
            template_html = _fallback_template()
        else:
            template_html = template_path.read_text(encoding="utf-8")

    # Inject decisions
    final_html = _inject_into_template(template_html, decisions, image, client_name, logo_b64)

    # Generate output path
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_path = output_dir / f"{client_name.lower()}_{timestamp}.png"

    # Render with Playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        page = await browser.new_page(viewport={"width": 1080, "height": 1080})

        await page.set_content(final_html, wait_until="networkidle")

        # Extra wait for Google Fonts
        await page.wait_for_timeout(1000)

        await page.screenshot(
            path=str(output_path),
            type="png",
            clip={"x": 0, "y": 0, "width": 1080, "height": 1080},
        )

        await browser.close()

    logger.info(f"Rendered: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")

    return RenderResult(
        final_image_path=str(output_path),
        width=1080,
        height=1080,
    )


def _fallback_template() -> str:
    """Minimal fallback template when no generated templates exist yet."""
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      width: 1080px; height: 1080px;
      background-color: var(--color-bg, #F5F4F0);
      font-family: var(--font-headline, 'Inter'), sans-serif;
      overflow: hidden;
      position: relative;
    }
    .hero-image {
      position: absolute;
      width: 100%; height: 100%;
      object-fit: var(--image-object-fit, contain);
      padding: var(--image-padding, 100px);
    }
    .text-zone {
      position: absolute;
      bottom: var(--headline-margin-y, 64px);
      left: var(--headline-margin-x, 64px);
      max-width: var(--headline-max-width, 75%);
      z-index: 10;
    }
    .headline {
      font-family: var(--font-headline, 'Inter'), sans-serif;
      font-size: var(--font-headline-size, 68px);
      font-weight: var(--font-headline-weight, 800);
      color: var(--color-text, #2A2A2A);
      line-height: var(--font-headline-line-height, 1.05);
      letter-spacing: var(--font-headline-tracking, -0.02em);
      text-transform: var(--font-headline-case, uppercase);
    }
    .subtext {
      font-family: var(--font-subtext, 'DM Sans'), sans-serif;
      font-size: var(--font-subtext-size, 18px);
      font-weight: var(--font-subtext-weight, 400);
      color: var(--color-subtext, #6B6B6B);
      margin-top: 16px;
      line-height: 1.5;
    }
    .client-label {
      position: absolute;
      bottom: 40px; right: 48px;
      font-family: var(--font-subtext, 'DM Sans'), sans-serif;
      font-size: 14px;
      font-weight: 600;
      color: var(--client-color, #C4A77D);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      opacity: 0.7;
    }
  </style>
</head>
<body>
  <img class="hero-image" src="{{IMAGE_PATH}}" alt="">
  <div class="text-zone">
    <div class="headline">{{HEADLINE}}</div>
    <div class="subtext">{{SUBTEXT}}</div>
  </div>
  <div class="client-label">{{CLIENT_NAME}}</div>
</body>
</html>"""
