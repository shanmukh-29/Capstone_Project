"""Offline-first Zepto policy assistant with a graph-shaped workflow and FastAPI."""
from pathlib import Path
import os
import re
from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    StateGraph = None
    START = END = None

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
KEYWORDS = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")
PROMPT_TEMPLATE = """Role: You are a Zepto policy support assistant.\nContext: {context}\nTask: Answer the user's question using only the context.\nFormat: Return JSON with answer, sources, and confidence.\nLength: Keep the answer under 80 words.\nConstraint: Do not answer using information not present in the provided context.\nFew-shot: Q: What are support hours? A: Support is available through in-app chat 24/7.\nQuestion: {question}"""


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    context: list[dict]
    answer: str
    sources: list[str]
    confidence: float


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


def load_documents():
    documents = []
    for path in sorted(DOCS.glob("*.txt")):
        documents.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
    return documents


def classify_intent(state: GraphState) -> GraphState:
    state["intent"] = "policy_question" if any(word in state["query"].lower() for word in KEYWORDS) else "general_question"
    return state


def retrieve_and_answer(state: GraphState) -> GraphState:
    query_words = set(re.findall(r"[a-z]+", state["query"].lower()))
    scored = []
    for document in load_documents():
        score = len(query_words.intersection(set(re.findall(r"[a-z]+", document["text"].lower()))))
        scored.append((score, document))
    context = [document for _, document in sorted(scored, key=lambda item: item[0], reverse=True)[:3]]
    top = context[0]["text"] if context else "No matching policy context was found."
    state["context"] = context
    state["answer"] = f"Based on the retrieved context: {top[:200]}"
    state["sources"] = [document["id"] for document in context]
    state["confidence"] = 1.0
    return state


def direct_answer(state: GraphState) -> GraphState:
    state["answer"] = "I can only answer questions about Zepto policies right now."
    state["sources"] = []
    state["confidence"] = 1.0
    return state


def route_after_classification(state: GraphState):
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    if StateGraph is None:
        return None
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_classification)
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


GRAPH = build_graph()


def run_graph(query: str) -> AskResponse:
    state: GraphState = {"query": query}
    if GRAPH is not None:
        state = GRAPH.invoke(state)
    else:
        classify_intent(state)
        if state["intent"] == "policy_question":
            retrieve_and_answer(state)
        else:
            direct_answer(state)
    return AskResponse(answer=state["answer"], sources=state["sources"], confidence=state["confidence"])


app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return run_graph(request.query)


if __name__ == "__main__":
    print(run_graph("What is the delivery policy?").model_dump_json())
    print(run_graph("What is the weather?").model_dump_json())
