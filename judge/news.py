import logging

from exa_py import Exa

from config import EXA_API_KEY, NEWS_DOMAINS, NEWS_MAX_AGE_HOURS, NEWS_TOP_N

logger = logging.getLogger(__name__)


def search_news(company_name: str, max_age_hours: int = NEWS_MAX_AGE_HOURS,
                top_n: int = NEWS_TOP_N) -> list[dict]:
    if not EXA_API_KEY:
        logger.error("exa api key not configured")
        return []

    try:
        exa = Exa(api_key=EXA_API_KEY)
        result = exa.search(
            query=f"{company_name} stock NSE",
            num_results=top_n,
            include_domains=NEWS_DOMAINS,
            contents={
                "highlights": True,
                "max_age_hours": max_age_hours,
            },
        )
    except Exception:
        logger.error("exa search failed for %s", company_name, exc_info=True)
        return []

    articles = []
    for r in result.results:
        snippet = ""
        if r.highlights:
            snippet = r.highlights[0]
        articles.append({
            "title": r.title or "",
            "snippet": snippet,
            "published_at": r.published_date or "",
        })

    if not articles:
        logger.warning("no news results for %s", company_name)

    return articles
