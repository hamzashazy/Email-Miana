import json
import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from .config import API_KEY, MODEL, BASE_URL, CATEGORIES, MAX_RETRIES, REQUEST_TIMEOUT, MAX_CONCURRENT_REQUESTS
from .prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None
_system_prompt_cache: Optional[str] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )
    return _client


async def classify_email(
    subject: str,
    snippet: str,
    from_email: str = "",
    property_address: str = "",
    offer_type: str = "",
    offer_price: str = "",
) -> dict:
    """Classify a single email using the LLM. Returns dict with classification, confidence, reasoning."""
    global _system_prompt_cache
    client = get_client()
    if _system_prompt_cache is None:
        _system_prompt_cache = build_system_prompt()
    user_prompt = build_user_prompt(subject, snippet, from_email, property_address, offer_type, offer_price)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _system_prompt_cache},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)

            classification = result.get("classification", "").lower().strip()
            if classification not in CATEGORIES:
                logger.warning("LLM returned invalid category '%s', falling back to needs_review", classification)
                classification = "needs_review"

            return {
                "classification": classification,
                "confidence": float(result.get("confidence", 0.0)),
                "reasoning": result.get("reasoning", ""),
                "raw_response": raw,
            }

        except json.JSONDecodeError as e:
            logger.warning("Attempt %d: JSON parse error: %s", attempt, e)
        except Exception as e:
            logger.warning("Attempt %d: API error: %s", attempt, e)

        if attempt < MAX_RETRIES:
            await asyncio.sleep(2 ** attempt)

    return {
        "classification": "needs_review",
        "confidence": 0.0,
        "reasoning": "Classification failed after retries",
        "raw_response": "",
    }


async def classify_batch(
    emails: list[dict],
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
) -> list[dict]:
    """Classify a batch of emails concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(emails)

    async def _classify_one(email: dict, idx: int) -> dict:
        async with semaphore:
            if idx > 0 and idx % 50 == 0:
                logger.info("Progress: %d/%d emails classified", idx, total)
            result = await classify_email(
                subject=email.get("subject", ""),
                snippet=email.get("snippet", ""),
                from_email=email.get("from_email", ""),
                property_address=email.get("property_address", ""),
                offer_type=email.get("offer_type", ""),
                offer_price=str(email.get("offer_price", "")),
            )
            return {**email, **result}

    tasks = [_classify_one(email, i) for i, email in enumerate(emails)]
    return await asyncio.gather(*tasks)
