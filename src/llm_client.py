"""
LLM Client — Claude Sonnet 4.6 via Anthropic API
"""

import os
import anthropic

_client = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    return _client

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 512   # Keep concise for voice output

# Cap how much conversation history is replayed to the model so token
# usage stays bounded on long sessions.
MAX_HISTORY_MESSAGES = 12


# ─── System Prompts ────────────────────────────────────────────────────────

EXPLAIN_PROMPT = """\
You are an expert AI tutor helping a student understand academic papers.

Rules:
1. Answer ONLY from the provided context. If the answer isn't there, say:
   "I don't have information about that in the current knowledge base."
2. Explain concepts using clear analogies and examples — don't just quote the text.
3. Keep responses to 3-4 sentences — optimised for voice playback.
4. End with the source paper name (e.g. "This comes from arXiv:2305.XXXXX").
5. If the student seems confused, offer to explain a different way.\
"""

QUIZ_PROMPT = """\
You are an AI tutor in quiz mode. Test the student's understanding.

Rules:
1. Generate ONE clear question based on the provided context.
2. If the student has already answered, give constructive feedback and explain
   the correct answer using the context.
3. Keep questions answerable from the provided context.
4. Vary types: conceptual ("What is X?"), application ("How would X be used?"),
   comparison ("What's the difference between X and Y?").
5. Keep responses to 3-4 sentences for voice playback.\
"""

FREE_PROMPT = """\
You are a friendly AI tutor. Answer the student's questions naturally.

Rules:
1. Prefer answering from the context, but you may use general knowledge
   when the context doesn't cover the question.
2. Be conversational and encouraging.
3. Keep responses to 3-4 sentences for voice playback.
4. Cite sources when quoting the context.\
"""

_PROMPTS = {
    "explain": EXPLAIN_PROMPT,
    "quiz":    QUIZ_PROMPT,
    "free":    FREE_PROMPT,
}


def _trim_to_sentence(text: str) -> str:
    """Cut a truncated response back to its last complete sentence so the
    voice output doesn't stop mid-thought."""
    end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if end > 0:
        return text[:end + 1]
    return text


# ─── Main function ─────────────────────────────────────────────────────────

def get_teaching_response(query: str, context: str, mode: str = "explain",
                          history: list | None = None) -> str:
    """
    Generate a teaching response from Claude.

    Args:
        query:   The student's question
        context: Retrieved passages from the knowledge base
        mode:    'explain' | 'quiz' | 'free'
        history: Prior conversation turns as [{"role", "content"}, ...]
                 (quiz feedback and follow-up questions need them)

    Returns:
        Response text string
    """
    system = _PROMPTS.get(mode, EXPLAIN_PROMPT)

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in (history or [])[-MAX_HISTORY_MESSAGES:]
        if msg.get("role") in ("user", "assistant") and msg.get("content")
    ]

    user_message = f"""Context from knowledge base:
---
{context}
---

Student's question: {query}"""

    messages.append({"role": "user", "content": user_message})

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages
    )

    text = next((b.text for b in response.content if b.type == "text"), "")

    if response.stop_reason == "max_tokens":
        text = _trim_to_sentence(text)

    return text
