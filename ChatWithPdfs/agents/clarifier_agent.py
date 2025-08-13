import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def _get_llm():
    model = os.getenv("CLARIFIER_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


_CLARIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a query clarity checker. If the question is underspecified or ambiguous, write ONE clarifying question. If it is clear enough, just answer with OK."),
    ("human", "{question}")
])

def get_clarification(question: str) -> Optional[str]:
    try:
        llm = _get_llm()
        msg = _CLARIFY_PROMPT.format_messages(question=question)
        out = llm.invoke(msg).content.strip()
        if out == "OK":
            return None
        if not out.endswith("?"):
            out = out + "?"
        return out
    except Exception:        
        return None
