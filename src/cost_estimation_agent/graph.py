from langgraph.graph import StateGraph, START, END

from .state import EstimationState
from .nodes import (
    ingestion,
    extractor,
    orchestrator,
    human_review,
    calculator,
    presentation,
)


def _need_human_router(state: EstimationState):
    route = "HUMAN" if state.get("needs_human") else "AUTO"
    print(f"[router] orchestrator -> {route}")
    return route


def build_graph() -> StateGraph:
    g = StateGraph(EstimationState)

    # ノード登録
    g.add_node("ingestion", ingestion.ingestion_node)
    g.add_node("extractor", extractor.extractor_node)
    g.add_node("orchestrator", orchestrator.orchestrator_node)
    g.add_node("human_review", human_review.human_in_the_loop_node)
    g.add_node("calculator", calculator.calculator_node)
    g.add_node("presentation", presentation.presentation_node)

    # エッジ設定
    g.add_edge(START, "ingestion")
    g.add_edge("ingestion", "extractor")
    g.add_edge("extractor", "orchestrator")

    g.add_conditional_edges(
        "orchestrator",
        _need_human_router,
        {
            "HUMAN": "human_review",
            "AUTO": "calculator",
        },
    )

    g.add_edge("human_review", "calculator")
    g.add_edge("calculator", "presentation")
    g.add_edge("presentation", END)

    return g


def compile_app(checkpointer=None):
    # 実装では checkpointer を設定推奨
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)


__all__ = ["build_graph", "compile_app"]
