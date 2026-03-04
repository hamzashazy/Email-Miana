from .config import CATEGORIES, CATEGORY_DESCRIPTIONS


def build_system_prompt() -> str:
    category_block = "\n".join(
        f"- **{cat}**: {CATEGORY_DESCRIPTIONS[cat]}" for cat in CATEGORIES
    )

    return f"""\
You are an expert email classifier for a real estate investment company called Miana.

Miana sends offers (cash offers, combo offers with seller financing, LOIs) to property sellers \
and their listing agents. You classify the REPLY emails that come back.

## Categories

{category_block}

## Rules

1. Classify based on the INTENT of the reply, not just individual words.
2. A "rejection" is any clear refusal — even polite ones like "thanks but no thanks."
3. Bounce-back / delivery failure / undeliverable emails count as "rejection."
4. A "counter_offer" MUST mention a specific dollar amount or concrete alternative terms from the seller side.
5. "interested" means genuine forward momentum (accepting, presenting to seller, scheduling next steps).
6. "replies" is for neutral acknowledgements, or clarifying questions with no clear stance.
7. Use "needs_review" ONLY when the email is truly ambiguous or you cannot confidently pick another category.
8. If the email contains multiple signals, pick the DOMINANT intent.

## Edge Cases

- "No seller financing" / "cash only" → rejection
- "Offer received, will present to seller" → interested
- "Thanks for the offer" (nothing more) → replies
- "Seller counters at $130K" → counter_offer
- "Can you resend the attachment?" → replies
- Delivery Status Notification / bounce → rejection

## Output Format

Respond with ONLY a JSON object — no markdown, no explanation:

{{"classification": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


def build_user_prompt(subject: str, snippet: str, from_email: str = "", property_address: str = "", offer_type: str = "", offer_price: str = "") -> str:
    parts = [f"Subject: {subject}", f"Body: {snippet}"]

    if from_email:
        parts.insert(0, f"From: {from_email}")
    if property_address:
        parts.append(f"Property: {property_address}")
    if offer_type:
        parts.append(f"Offer type: {offer_type}")
    if offer_price:
        parts.append(f"Offer price: ${offer_price}")

    return "\n".join(parts)
