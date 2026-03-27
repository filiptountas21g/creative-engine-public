"""Step 1: Research — Perplexity Sonar finds what's trending."""

import logging
import httpx

import config
from pipeline.types import PipelineInput, ResearchResult

logger = logging.getLogger(__name__)

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"


async def research(input: PipelineInput) -> ResearchResult:
    """
    Call Perplexity Sonar to find trending topics for the client's industry.

    Input: PipelineInput (client, brief, platform)
    Output: ResearchResult (trends text)
    """
    if not config.PERPLEXITY_API_KEY:
        logger.warning("No PERPLEXITY_API_KEY set — skipping research step")
        return ResearchResult(trends="No research data available (API key not set).")

    query = (
        f"What are the top 3 trends in the industry related to '{input.brief}' "
        f"this week, relevant to social media content in Greece? "
        f"Include any relevant news, competitor activity, or cultural moments. "
        f"Be specific and current. Keep it under 300 words."
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                PERPLEXITY_URL,
                headers={
                    "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": query}],
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            logger.info(f"Research: got {len(content)} chars from Perplexity")

            return ResearchResult(
                trends=content,
                raw_response=str(data),
            )

    except Exception as e:
        logger.error(f"Perplexity research failed: {e}")
        return ResearchResult(
            trends=f"Research unavailable: {str(e)[:100]}",
        )
