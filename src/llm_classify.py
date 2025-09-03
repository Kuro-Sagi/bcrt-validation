# src/llm_classify.py
import os
from typing import Dict, Optional

def classify_with_llm(item_id: str, prompt_text: str, answer_text: str, rubric: Dict) -> Optional[str]:
    """
    Return one of {"correct","intuitive","other"} using OpenAI Chat Completions.
    Defaults to GPT-5 nano, or respects OPENAI_MODEL / OPENAI_MODEL_SNAPSHOT.
    Returns None if no API key or if the call fails.
    """
    try:
        from openai import OpenAI
    except Exception:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", os.environ.get("OPENAI_MODEL_SNAPSHOT", "gpt-5-nano"))

    sys_prompt = (
        "You are a strict classifier for CRT-style free-text answers.\n"
        "Output exactly one label on a single line: correct, intuitive, or other.\n"
        "Use this rubric: 'correct' strings/phrases/numbers, and 'intuitive' lure strings/phrases.\n"
        "Allow paraphrases and numeric equivalents. If the answer clearly states the right idea,\n"
        "label correct; if it clearly gives the common fast-but-wrong lure, label intuitive; else other."
    )
    user_prompt = (
        f"ITEM_ID: {item_id}\n"
        f"QUESTION: {prompt_text}\n"
        f"ANSWER: {answer_text}\n"
        f"RUBRIC: {rubric}\n"
        "Return only one of: correct | intuitive | other"
    )

    try:
        resp = client.chat.completions.create(
            model=model,  # GPT-5 nano by default
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label in {"correct", "intuitive", "other"}:
            return label
        return None
    except Exception:
        return None