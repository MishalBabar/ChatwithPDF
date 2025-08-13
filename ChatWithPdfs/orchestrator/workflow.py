import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents.clarifier_agent import get_clarification
from agents.rag_agent import answer_from_pdfs
from agents.web_search_agent import search_web

ALLOW_WEB_FALLBACK = os.getenv("ALLOW_WEB_FALLBACK", "false").lower() in {"1","true","yes"}


def _explicit_web_request(q: str) -> bool:
    ql = q.lower()
    triggers = ["search online", "search the web", "web search", "lookup online", "google this", "browse"]
    return any(t in ql for t in triggers)


class State(TypedDict, total=False):
    question: str
    history: List[str]
    final_answer: str
    route: str  


def entry_node(state: State) -> State:
    q = state["question"]
    if _explicit_web_request(q):
        state["route"] = "web"
        return state
    state["route"] = "clarify"
    return state


def clarify_node(state: State) -> State:
    q = state["question"]
    clarification = get_clarification(q)
    if clarification:
        state["final_answer"] = clarification
        state["route"] = "clarify"
    else:
        state["route"] = "pdf"
    return state


def rag_node(state: State) -> State:
    q = state["question"]
    rag = answer_from_pdfs(q)
    hits = rag.get("hits", 0)
    answer = rag.get("answer", "").strip()
    if hits > 0:
        srcs = rag.get("sources", [])
        suffix = ("Sources:" + "".join(f"- {s}" for s in srcs)) if srcs else ""
        state["final_answer"] = (answer or "(Found relevant passages; answer not explicit. See sources)") + suffix
        state["route"] = "pdf"
        return state

    # No hits in PDFs
    if ALLOW_WEB_FALLBACK:
        state["route"] = "web"
    else:
        state["final_answer"] = ("Couldn't find this in the provided PDFs")
        state["route"] = "pdf"
    return state


def web_node(state: State) -> State:
    q = state["question"]
    web = search_web(q)
    state["final_answer"] = ("Here are web results (fallback):" + web)
    state["route"] = "web"
    return state


# Build LangGraph
_graph = StateGraph(State)
_graph.add_node("entry", entry_node)
_graph.add_node("clarify", clarify_node)
_graph.add_node("rag", rag_node)
_graph.add_node("web", web_node)

_graph.set_entry_point("entry")
_graph.add_edge("entry", "clarify")
_graph.add_edge("clarify", "rag")
_graph.add_conditional_edges("rag",lambda s: "web" if s.get("route") == "web" else END,{"web": "web", END: END})
_graph.add_edge("web", END)

compiled_graph = _graph.compile()


class Orchestrator:
    def __init__(self):
        self.history: List[str] = []

    def handle(self, user_question: str) -> str:
        self.history.append(user_question)
        state: State = {"question": user_question, "history": self.history}
        out = compiled_graph.invoke(state)
        return out.get("final_answer", "I wasn't able to produce an answer.")


# Singleton orchestrator for API
orchestrator = Orchestrator()