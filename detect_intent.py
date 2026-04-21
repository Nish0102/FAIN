import os
import re
import requests

# ── CONFIG ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

INTENTS = ["need_support", "need_relax", "need_rest", "play_music", "unknown"]

SYSTEM_PROMPT = """You are an intent classifier for a voice assistant.
Given a user's spoken message, return ONLY one of these intent labels:
- need_support   → user feels stressed, anxious, overwhelmed, sad, terrible
- need_relax     → user feels okay, fine, neutral, wants to chill
- need_rest      → user feels tired, sleepy, exhausted, drained
- play_music     → user wants music played
- unknown        → none of the above

Reply with ONLY the label. No explanation. No punctuation."""

# ── KEYWORD FALLBACK (runs if API is unavailable) ────────────────────────────
KEYWORD_MAP = {
    "need_support": ["terrible", "bad", "worst", "stress", "anxious",
                     "overwhelmed", "sad", "depressed", "upset", "nervous"],
    "need_relax":   ["alright", "okay", "fine", "normal", "chill",
                     "calm", "relax", "neutral", "decent"],
    "need_rest":    ["tired", "exhausted", "sleepy", "drained",
                     "fatigue", "sleepy", "drowsy", "worn out"],
    "play_music":   ["play music", "play a song", "put on music",
                     "play something", "music please"],
}

def _keyword_fallback(text: str) -> str:
    text = text.lower()
    for intent, keywords in KEYWORD_MAP.items():
        if any(kw in text for kw in keywords):
            return intent
    return "unknown"

# ── AI INTENT DETECTION ──────────────────────────────────────────────────────
def _detect_via_claude(text: str) -> str | None:
    if not ANTHROPIC_API_KEY:
        return None

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",  # fast + cheap for classification
                "max_tokens": 20,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": text}],
            },
            timeout=5,
        )
        response.raise_for_status()
        label = response.json()["content"][0]["text"].strip().lower()

        # Sanitize — only accept known intents
        label = re.sub(r"[^a-z_]", "", label)
        return label if label in INTENTS else None

    except Exception as e:
        print(f"[Intent] Claude API failed: {e}")
        return None

# ── PUBLIC FUNCTION ──────────────────────────────────────────────────────────
def detect_intent(text: str) -> str:
    if not text or not text.strip():
        return "unknown"

    intent = _detect_via_claude(text)

    if intent:
        print(f"[Intent] Claude detected: {intent}")
    else:
        intent = _keyword_fallback(text)
        print(f"[Intent] Keyword fallback: {intent}")

    return intent