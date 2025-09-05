from ..state import EstimationState


def ingestion_node(state: EstimationState) -> EstimationState:
    # 図面の読み込み/形式判定/ページ展開など最低限の前処理
    # state["input_doc"] は渡されている前提
    trace = state.get("trace", [])
    trace.append("ingestion")
    state["trace"] = trace
    rfq = state.get("meta", {}).get("rfq_id", "UNKNOWN")
    print(f"[node] ingestion: rfq={rfq} 入力は2D図面を想定")
    return state


__all__ = ["ingestion_node"]
