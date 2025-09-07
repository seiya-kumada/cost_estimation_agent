from ..state import EstimationState
from ..tools import gpt4o_extract_material_mass
from ._utils import record_node_trace


def extractor_node(state: EstimationState) -> EstimationState:
    # 抽出情報を格納する準備
    extracted = dict(state.get("extracted") or {})
    issues = []
    conf = 0.1

    try:
        doc = state.get("input_doc")
        if doc is None:
            raise KeyError("input_doc is missing in state")

        # 図面から材料と質量を抽出
        res = gpt4o_extract_material_mass(doc)
        material = res.get("material")
        mass_kg = res.get("mass_kg")

        # 抽出結果を state に格納
        extracted["material"] = material
        extracted["mass_kg"] = mass_kg
        extracted["llm_raw"] = res.get("raw")

        # 取得きなかった項目をissuesに追加する
        if not material:
            issues.append("material")
        if mass_kg is None:
            issues.append("mass_kg")

        # issuesが空なら高信頼(0.9)、1つなら中信頼(0.5)、2つなら低信頼(0.1)
        if not issues:
            conf = 0.9
        elif len(issues) == 1:
            conf = 0.5
    except Exception as e:
        issues = ["material", "mass_kg"]
        conf = 0.1
        print(f"[node] extractor: 例外により抽出失敗: {e}")

    # 履歴を残す。
    record_node_trace(state, "extractor")

    # stateを更新して返す
    state["extracted"] = extracted
    state["extraction_confidence"] = conf
    state["extraction_issues"] = issues
    print(
        f"[node] extractor: material={extracted.get('material')} mass_kg={extracted.get('mass_kg')} conf={conf:.2f} issues={issues}"
    )
    return state


__all__ = ["extractor_node"]
