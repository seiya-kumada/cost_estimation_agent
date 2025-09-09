from ..state import EstimationState
from ._utils import record_node_trace

# Confidence threshold to trigger human review (currently ignored)
CONFIDENCE_THRESHOLD = 0.95


def orchestrator_node(state: EstimationState) -> EstimationState:
    """抽出結果に関わらず、必ず人手確認を要求するノード。

    振る舞い:
    - 以前は `extraction_confidence` と `extraction_issues` に基づき分岐していましたが、
      現在は必ず `needs_human=True` を設定して HITL を実施します。
    - トレースに "orchestrator" を追加し、従来条件（low_conf/missing）も参考としてログ出力します。

    Args:
    - state: エージェントの共有状態。

    Returns:
    - EstimationState: `needs_human=True` を反映した状態。
    """
    # 従来条件を計算（ログ用）。判定自体は強制的に True。
    low_conf = state.get("extraction_confidence", 0) < CONFIDENCE_THRESHOLD
    missing = bool(state.get("extraction_issues"))
    state["needs_human"] = True
    record_node_trace(state, "orchestrator")
    print(
        f"[node] orchestrator: low_conf={low_conf} missing={missing} -> needs_human=True (forced)"
    )
    return state


__all__ = ["orchestrator_node"]
