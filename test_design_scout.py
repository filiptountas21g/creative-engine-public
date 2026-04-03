"""
Standalone test: Perplexity Design Scout
Searches for fresh design inspiration based on client industries.
Run: python3 test_design_scout.py
"""

import asyncio
import json
import logging
import httpx

import config
from brain.client import Brain

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


# ── Search queries by category ──────────────────────────────────────

def _build_scout_queries(industry: str, client_name: str) -> list[str]:
    """Generate 3 diverse search queries to find fresh design inspiration."""
    return [
        # 1. Industry-specific social media design trends
        (
            f"Best social media post designs for {industry} brands 2025-2026. "
            f"Show me award-winning or viral Instagram/Facebook posts from {industry} companies. "
            f"Focus on visual layout, composition, and creative design approaches. "
            f"Include specific examples with URLs if possible."
        ),
        # 2. Broader design trend scouting
        (
            f"Latest graphic design trends for social media 2026. "
            f"What new layout styles, typography trends, and visual compositions are trending? "
            f"Especially for brand posts, not memes. Think editorial, premium, modern. "
            f"Include examples from Behance, Dribbble, or brand accounts."
        ),
        # 3. Anti-repetition: unusual/experimental layouts
        (
            f"Most creative and unusual social media post layouts 2025-2026. "
            f"Asymmetric designs, overlapping elements, broken grids, collage style, "
            f"3D elements, kinetic typography, mixed media. "
            f"Not standard centered-text-over-image. Show bold experimental designs."
        ),
    ]


async def _perplexity_search(query: str, label: str) -> dict:
    """Run a single Perplexity search and return structured result."""
    if not config.PERPLEXITY_API_KEY:
        return {"label": label, "error": "No PERPLEXITY_API_KEY set"}

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                PERPLEXITY_URL,
                headers={
                    "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 2048,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            # Extract citations if available
            citations = data.get("citations", [])

            return {
                "label": label,
                "content": content,
                "citations": citations,
                "tokens_used": data.get("usage", {}),
            }

    except Exception as e:
        return {"label": label, "error": str(e)}


async def _extract_layout_tags(search_results: list[dict]) -> str:
    """Use Claude to extract actionable layout descriptions from search results.
    This is what would get stored in Brain for Opus to use during generation."""

    combined_text = ""
    for r in search_results:
        if "content" in r:
            combined_text += f"\n\n--- {r['label']} ---\n{r['content']}"

    # Use Anthropic to extract structured layout inspiration
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=(
            "You are a design analyst. Extract actionable layout inspiration from design research. "
            "For each interesting layout you find, describe it in a way that a template generator can use."
        ),
        messages=[{
            "role": "user",
            "content": f"""From this design research, extract 5-8 SPECIFIC layout ideas I haven't tried.

For each layout, give me:
1. **Name**: A short descriptive name (e.g. "Diagonal Split", "Floating Cards")
2. **Structure**: Where does the image go? Where does the text go? What's the grid?
3. **What makes it fresh**: Why is this different from a standard centered-text post?
4. **Best for**: What type of content suits this layout? (product, event, quote, team, etc.)

RESEARCH DATA:
{combined_text}

Return as a structured list. Be specific about positions, proportions, and visual elements."""
        }],
    )

    return response.content[0].text


async def run_scout(client_name: str = None, industry: str = None):
    """Run the full design scout pipeline."""

    print("\n" + "=" * 70)
    print("🔍 DESIGN SCOUT — Perplexity Fresh Layout Finder")
    print("=" * 70)

    # If no client specified, try to load from Brain
    if not client_name and not industry:
        try:
            brain = Brain()
            clients = brain.get_clients()
            if clients:
                print(f"\nFound {len(clients)} clients in Brain:")
                for i, c in enumerate(clients):
                    print(f"  {i+1}. {c['name']} ({c['industry']})")
                # Use first client as default
                client_name = clients[0]["name"]
                industry = clients[0]["industry"]
                print(f"\nUsing: {client_name} ({industry})")
            else:
                industry = "health and beauty"
                print(f"\nNo clients in Brain, using default industry: {industry}")
        except Exception:
            industry = "health and beauty"
            print(f"\nCouldn't read Brain, using default industry: {industry}")

    if not industry:
        industry = "general business"

    # Step 1: Build and run searches
    queries = _build_scout_queries(industry, client_name or "brand")

    labels = [
        f"🏢 Industry-specific ({industry})",
        "🎨 Broader design trends",
        "🧪 Experimental/unusual layouts",
    ]

    print(f"\n{'─' * 50}")
    print("STEP 1: Running 3 Perplexity searches...")
    print(f"{'─' * 50}")

    # Run all 3 searches in parallel
    tasks = [
        _perplexity_search(q, l)
        for q, l in zip(queries, labels)
    ]
    results = await asyncio.gather(*tasks)

    # Print raw results
    total_tokens = 0
    for r in results:
        print(f"\n{r['label']}:")
        if "error" in r:
            print(f"  ❌ Error: {r['error']}")
        else:
            content_preview = r["content"][:500]
            print(f"  {content_preview}...")
            if r["citations"]:
                print(f"\n  📎 Citations ({len(r['citations'])}):")
                for url in r["citations"][:5]:
                    print(f"     • {url}")
            tokens = r.get("tokens_used", {})
            used = tokens.get("total_tokens", 0)
            total_tokens += used
            print(f"  📊 Tokens: {used}")

    print(f"\n{'─' * 50}")
    print(f"Total Perplexity tokens: {total_tokens}")
    print(f"{'─' * 50}")

    # Step 2: Extract actionable layout tags
    valid_results = [r for r in results if "content" in r]
    if not valid_results:
        print("\n❌ No search results to analyze.")
        return

    print(f"\n{'─' * 50}")
    print("STEP 2: Extracting layout inspiration with Claude...")
    print(f"{'─' * 50}")

    layouts = await _extract_layout_tags(valid_results)
    print(f"\n{layouts}")

    # Step 3: Show what would be stored in Brain
    print(f"\n{'─' * 50}")
    print("STEP 3: What gets stored in Brain (for Opus to use)")
    print(f"{'─' * 50}")
    print("\nThis text would be saved as a 'design_scout' entry in Brain.")
    print("During generation, if anti-repetition detects staleness,")
    print("Opus would receive one of these layouts as a structural constraint.")
    print(f"\nEntry size: {len(layouts)} chars (~{len(layouts) // 4} tokens when read)")

    # Also show cost estimate
    print(f"\n{'─' * 50}")
    print("💰 COST ESTIMATE")
    print(f"{'─' * 50}")
    print(f"  Perplexity (3 searches): ~$0.01-0.03")
    print(f"  Claude Sonnet (extraction): ~$0.01-0.02")
    print(f"  Total scout run: ~$0.02-0.05")
    print(f"  If run weekly: ~$0.08-0.20/month")
    print(f"\n  Per-post cost impact: $0.00 (scout data reused from Brain)")


if __name__ == "__main__":
    asyncio.run(run_scout())
