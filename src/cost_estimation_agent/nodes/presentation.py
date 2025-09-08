from ..state import EstimationState
from ..tools import store_history
from ._utils import record_node_trace


def presentation_node(state: EstimationState) -> EstimationState:
    """見積結果を整形し、提示用ペイロードを作成・出力するノード。

    振る舞い:
    - `total_cost` と `cost_breakdown.material_pricing` から材料/単価/質量を取得
    - 見積結果をログに表示（数値フォーマットは可能なら小数2桁）
    - 失敗時は `errors` をまとめて理由をログ出力
    - `presentation_payload` を生成し、`store_history` に渡す（履歴保存はスタブ）
    - トレースに "presentation" を追加

    Args:
    - state: エージェントの共有状態。

    Returns:
    - EstimationState: 変更を反映した状態（`trace` と `presentation_payload` を更新）。
    """
    # 履歴を残す。
    record_node_trace(state, "presentation")

    # stateから情報を取得
    total_cost = state.get("total_cost")
    breakdown = state.get("cost_breakdown", {}) or {}
    mat = (breakdown.get("material_pricing") or {}) if isinstance(breakdown, dict) else {}

    # material_pricingの情報を取得
    material = mat.get("material")
    unit_price = mat.get("unit_price_kg")
    mass_kg = mat.get("mass_kg")
    errors = state.get("errors", []) or []

    if total_cost is not None:
        try:
            print(f"[presentation] 見積結果: {float(total_cost):.2f} 円")
        except Exception:
            print(f"[presentation] 見積結果: {total_cost} 円")
        print(f"  根拠: 材料={material}, 単価={unit_price} 円/kg, 質量={mass_kg} kg")
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
