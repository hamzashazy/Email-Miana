import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = "https://api.openai.com/v1"

CATEGORIES = [
    "rejection",
    "counter_offer",
    "interested",
    "replies",
    "needs_review",
]

CATEGORY_DESCRIPTIONS = {
    "rejection": (
        "The seller or agent has clearly declined, rejected, or refused the offer. "
        "Includes: not interested, declined, won't accept, no financing allowed, property sold, "
        "off market, do not contact, already under contract, offer too low, zero interest, waste of time, "
        "not my listing, wrong agent. Also includes bounce-back / delivery failure / undeliverable emails."
    ),
    "counter_offer": (
        "The seller or agent responds with a SPECIFIC dollar amount or concrete alternative terms. "
        "Includes: counter at $X, seller wants $X, asking price is $X, would you do $X cash, "
        "lowest they'd go is $X. Must have an actual number from the seller side."
    ),
    "interested": (
        "The seller or agent has ACCEPTED the offer or is explicitly moving forward. "
        "Includes: offer accepted, let's move forward, seller accepts, will present to seller, "
        "forwarding to seller, schedule a call/showing."
    ),
    "replies": (
        "A general reply that acknowledges the email but does NOT clearly accept, reject, or counter. "
        "Includes: thanks for the email, received/ok, attachment issues (can't see offer), "
        "out of office, noted/acknowledged, sorry for delay, asking clarifying questions, "
        "proof of funds requests, greeting responses."
    ),
    "needs_review": (
        "The email is ambiguous, unclear, or doesn't fit neatly into the other categories. "
        "A human should review this email to determine the correct classification. "
        "Use this ONLY when none of the other categories clearly apply."
    ),
}

MAX_CONCURRENT_REQUESTS = 8
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
