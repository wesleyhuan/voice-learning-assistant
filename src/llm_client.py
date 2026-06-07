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

MODEL     = "claude-sonnet-4-6"
MAX_TOKENS = 512   # Keep concise for voice output


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


# ─── Main function ─────────────────────────────────────────────────────────

def get_teaching_response(query: str, context: str, mode: str = "explain") -> str:
    """
    Generate a teaching response from Claude.

    Args:
        query:   The student's question
        context: Retrieved passages from the knowledge base
        mode:    'explain' | 'quiz' | 'free'

    Returns:
        Response text string
    """
    system = _PROMPTS.get(mode, EXPLAIN_PROMPT)

    user_message = f"""Context from knowledge base:
---
{context}
---

Student's question: {query}"""

    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_message}]
    )

    return response.content[0].text
