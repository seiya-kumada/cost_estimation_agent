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
    """orchestratorの判定結果から次の遷移先を返すルータ。

    振る舞い:
    - `state["needs_human"]` が真なら "HUMAN"、偽/未設定なら "AUTO" を返す。
    - ルーティング結果をログに出力する。

    Args:
    - state: エージェントの共有状態。

    Returns:
    - str: 遷移ラベル（"HUMAN" | "AUTO"）。
    """
    route = "HUMAN" if state.get("needs_human") else "AUTO"
    print(f"[router] orchestrator -> {route}")
    return route


def build_graph() -> StateGraph:
    """見積ワークフローのノードと遷移を定義してグラフを構築する。

    振る舞い:
    - ノード登録: ingestion → extractor → orchestrator → (human_review?) → calculator → presentation
    - 条件分岐: orchestrator の判定を `_need_human_router` で評価し、"HUMAN" なら human_review、"AUTO" なら calculator へ遷移
    - 始端/終端: START から ingestion へ、presentation から END へ接続

    Returns:
    - StateGraph: 構築済みの状態グラフ（未コンパイル）
    """
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
    """LangGraph をコンパイルし、実行可能なアプリを返す。

    振る舞い:
    - `build_graph()` で見積フローのグラフを構築
    - `checkpointer` を指定すると状態永続化・再開を有効化（実装に依存）
    - `graph.compile(checkpointer=...)` を呼び出してコンパイル済みの実行体を返す

    Args:
    - checkpointer: LangGraph 互換のチェックポインタ（省略可）

    Returns:
    - コンパイル済みのグラフ実行体（呼び出し可能オブジェクト）
    """
    # 実装では checkpointer を設定推奨
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)


__all__ = ["build_graph", "compile_app"]
