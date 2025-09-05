from ..state import EstimationState
from ..tools import store_history


def presentation_node(state: EstimationState) -> EstimationState:
    # 価格提示（ユーザ向け表示）
    trace = state.get("trace", [])
    trace.append("presentation")
    state["trace"] = trace

    total_cost = state.get("total_cost")
    breakdown = state.get("cost_breakdown", {}) or {}
    mat = (breakdown.get("material_pricing") or {}) if isinstance(breakdown, dict) else {}

    material = mat.get("material")
    unit_price = mat.get("unit_price_kg")
    mass_kg = mat.get("mass_kg")
    errors = state.get("errors", []) or []

    if total_cost is not None:
        try:
            print(f"[presentation] 見積結果: {float(total_cost):.2f} 円")
        except Exception:
            print(f"[presentation] 見積結果: {total_cost} 円")
        print(
            f"  根拠: 材料={material}, 単価={unit_price} 円/kg, 質量={mass_kg} kg"
        )
        message = "見積結果を提示しました。"
    else:
        reason = ", ".join(map(str, errors)) if errors else "原因不明"
        print(f"[presentation] 見積不能: {reason}")
        message = f"見積不能: {reason}"

    state["presentation_payload"] = {
        "summary": {
            "message": message,
            "total_cost": total_cost,
        },
        "material_pricing": {
            "material": material,
            "unit_price_kg": unit_price,
            "mass_kg": mass_kg,
        },
        "errors": errors,
    }
    # 履歴保存（実装はツール側のスタブ。後続で具体化可）
    store_history(state["presentation_payload"])  # noqa: F841 - placeholder
    return state


__all__ = ["presentation_node"]
