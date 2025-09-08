from ..state import EstimationState
from ._utils import record_node_trace

# Confidence threshold to trigger human review
CONFIDENCE_THRESHOLD = 0.95


def orchestrator_node(state: EstimationState) -> EstimationState:
    """抽出結果を評価し、人手確認の要否を決定するノード。

    振る舞い:
    - `extraction_confidence` と `extraction_issues` を参照して Need-HITL を判定
      - しきい値: `extraction_confidence < CONFIDENCE_THRESHOLD` または `extraction_issues` が非空なら `needs_human=True`
    - 判定結果を `state["needs_human"]` に設定
    - トレースに "orchestrator" を追加し、判定内容をログ出力

    Args:
    - state: エージェントの共有状態。

    Returns:
    - EstimationState: 変更を反映した状態（`needs_human` と `trace` が更新されます）。
    """
    # しきい値判定＆人手確認の要否（閾値は CONFIDENCE_THRESHOLD）
    low_conf = state.get("extraction_confidence", 0) < CONFIDENCE_THRESHOLD
    missing = bool(state.get("extraction_issues"))  # listが空でなければTrue
    state["needs_human"] = low_conf or missing
    record_node_trace(state, "orchestrator")
    print(f"[node] orchestrator: low_conf={low_conf} missing={missing} -> needs_human={state['needs_human']}")
    return state


__all__ = ["orchestrator_node"]
