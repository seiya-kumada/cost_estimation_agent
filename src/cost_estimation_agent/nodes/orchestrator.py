from ..state import EstimationState


def orchestrator_node(state: EstimationState) -> EstimationState:
    # しきい値判定＆人手確認の要否（閾値を 0.95 に強化）
    low_conf = state.get("extraction_confidence", 0) < 0.95
    missing = bool(state.get("extraction_issues"))
    state["needs_human"] = low_conf or missing
    trace = state.get("trace", [])
    trace.append("orchestrator")
    state["trace"] = trace
    print(
        f"[node] orchestrator: low_conf={low_conf} missing={missing} -> needs_human={state['needs_human']}"
    )
    return state


__all__ = ["orchestrator_node"]
