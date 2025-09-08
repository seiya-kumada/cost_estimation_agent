from ..state import EstimationState
from ._utils import record_node_trace


def ingestion_node(state: EstimationState) -> EstimationState:
    """図面入力を受け取り、トレース記録などの軽微な前処理を行うノード。

    振る舞い:
    - 前提: `state["input_doc"]` が与えられている（画像バイト列または画像パスを想定）
    - 将来の拡張: 読み込み/形式判定/ページ展開などの前処理をここで実施予定（現状は通過処理）
    - トレースに "ingestion" を追加し、`meta.rfq_id` をログ出力

    Args:
    - state: エージェントの共有状態。

    Returns:
    - EstimationState: 変更を反映した状態（`trace` が更新されます）。
    """
    record_node_trace(state, "ingestion")
    rfq = state.get("meta", {}).get("rfq_id", "UNKNOWN")
    print(f"[node] ingestion: rfq={rfq} 入力は2D図面を想定")
    return state


__all__ = ["ingestion_node"]
