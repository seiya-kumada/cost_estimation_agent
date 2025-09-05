from typing import Any, Dict

from ..state import EstimationState
from ..tools import materials_db_query
from ._utils import record_node_trace


def _calc_material_cost_and_errors(
    unit_price: float | None,
    mass_kg: float | None,
    material: str | None,
    errors: list[str],
) -> tuple[float | None, float | None, list[str]]:
    """材料コストと合計の計算、ならびに不足時のエラー付与を行う。

    引数:
    - unit_price: 単価 (JPY/kg)。見つからない場合は None。
    - mass_kg: 質量 (kg)。未取得の場合は None。
    - material: 材料名。未取得の場合は None。
    - errors: これまでのエラー配列（追記用にコピーされます）。

    戻り値:
    - material_cost: 材料コスト (JPY) または None。
    - total_cost: 合計コスト (JPY) または None（現状は材料コストに等しい）。
    - errors: 追記後のエラー配列。
    """
    out_errors = list(errors)
    material_cost: float | None = None
    total_cost: float | None = None

    if unit_price is not None and isinstance(mass_kg, (int, float)):
        material_cost = float(unit_price) * float(mass_kg)
        total_cost = material_cost
    else:
        if material is None:
            out_errors.append("material_not_specified")
        if mass_kg is None:
            out_errors.append("mass_not_specified")
        if unit_price is None and material is not None:
            out_errors.append(f"unit_price_not_found:{material}")

    return material_cost, total_cost, out_errors


def calculator_node(state: EstimationState) -> EstimationState:
    # 履歴を残す。
    record_node_trace(state, "calculator")

    # 抽出情報を取得
    extracted = state.get("extracted", {})
    material = extracted.get("material")
    mass_kg = extracted.get("mass_kg")

    # エラー履歴リストをコピーする。
    errors = list(state.get("errors", []) or [])

    # 材料単価をDBから取得
    mat_info = materials_db_query(material)
    unit_price = mat_info.get("unit_price_kg") if mat_info.get("found") else None

    # コスト計算とエラー更新
    material_cost, total_cost, errors = _calc_material_cost_and_errors(unit_price, mass_kg, material, errors)

    # 内訳を構築
    breakdown = make_cost_breakdown(
        mat_info=mat_info,
        extracted_material=material,
        unit_price=unit_price,
        mass_kg=mass_kg,
        material_cost=material_cost,
    )

    # 状態を更新して返す
    state["errors"] = errors
    state["cost_breakdown"] = breakdown
    state["total_cost"] = total_cost

    if total_cost is not None:
        print(
            f"[node] calculator: material={breakdown['material_pricing']['material']} unit={unit_price} mass={mass_kg} total={total_cost}"
        )
    else:
        print(f"[node] calculator: 見積不能（不足/未登録）。errors={errors}")
    return state


def make_cost_breakdown(
    *,
    mat_info: Dict[str, Any],
    extracted_material: str | None,
    unit_price: float | None,
    mass_kg: float | None,
    material_cost: float | None,
) -> Dict[str, Any]:
    """材料原価に関するブレークダウン(dict)を組み立てる。

    引数:
    - mat_info: `materials_db_query` の結果辞書。
    - extracted_material: 抽出またはHITLで得た材料名（DB未ヒット時のフォールバック表示用）。
    - unit_price: 単価 (JPY/kg)。
    - mass_kg: 質量 (kg)。
    - material_cost: 材料コスト (JPY)。

    戻り値:
    - `Dict[str, Any]`: `state['cost_breakdown']` に格納する構造。
    """
    material_display = mat_info.get("name") if mat_info.get("found") else extracted_material
    return {
        "material_pricing": {
            "material": material_display,
            "unit_price_kg": unit_price,
            "mass_kg": mass_kg,
            "material_cost": material_cost,
            "source": mat_info.get("source"),
        }
    }


__all__ = ["calculator_node", "make_cost_breakdown"]
