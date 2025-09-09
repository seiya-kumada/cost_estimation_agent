from ..state import EstimationState
from ..tools import extract_material_mass_via_gpt4o
from ._utils import record_node_trace


def extractor_node(state: EstimationState) -> EstimationState:
    """図面から材料と質量(kg)を抽出し、状態へ反映するノード。

    振る舞い:
    - `input_doc`（画像バイト列または画像パス）を入力に GPT-4o で material/mass_kg を抽出
    - `extracted` に material・mass_kg・llm_raw を格納
    - 未取得項目を `extraction_issues` に記録し、`extraction_confidence` を設定（高/中/低）
    - トレースに "extractor" を追加

    Args:
    - state: エージェントの状態。

    Returns:
    - EstimationState: 抽出結果・信頼度・issues を反映した状態。
    """
    # 抽出情報を格納する準備
    extracted = dict(state.get("extracted") or {})
    issues = []
    conf = 0.1

    try:
        doc = state.get("input_doc")
        if doc is None:
            raise KeyError("input_doc is missing in state")

        # 図面から材料と質量を抽出
        res = extract_material_mass_via_gpt4o(doc)
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
        "[node] extractor: "
        f"material={extracted.get('material')} "
        f"mass_kg={extracted.get('mass_kg')} "
        f"conf={conf:.2f} "
        f"issues={issues}"
    )
    return state


__all__ = ["extractor_node"]
