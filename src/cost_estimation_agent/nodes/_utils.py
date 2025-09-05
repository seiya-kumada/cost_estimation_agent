from ..state import EstimationState


def record_node_trace(state: EstimationState, node: str) -> None:
    """Append the given node name to the state's trace list.

    Creates the list if missing and updates in place.
    """
    trace = state.get("trace", [])
    trace.append(node)
    state["trace"] = trace


__all__ = ["record_node_trace"]

