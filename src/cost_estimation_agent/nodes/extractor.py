from ..state import EstimationState
from ..tools import gpt4o_extract_material_mass


def extractor_node(state: EstimationState) -> EstimationState:
    # GPT-4o（Structured Output）で材料・質量(kg)を抽出する
    extracted = dict(state.get("extracted") or {})
    issues = []
    conf = 0.1

    try:
        res = gpt4o_extract_material_mass(state["input_doc"])
        material = res.get("material")
        mass_kg = res.get("mass_kg")
        extracted["material"] = material
        extracted["mass_kg"] = mass_kg
        extracted["llm_raw"] = res.get("raw")

        if not material:
            issues.append("material")
        if mass_kg is None:
            issues.append("mass_kg")

        if not issues:
            conf = 0.9
        elif len(issues) == 1:
            conf = 0.5
    except Exception as e:
        issues = ["material", "mass_kg"]
        conf = 0.1
        print(f"[node] extractor: 例外により抽出失敗: {e}")

    trace = state.get("trace", [])
    trace.append("extractor")
    state["trace"] = trace

    state["extracted"] = extracted
    state["extraction_confidence"] = conf
    state["extraction_issues"] = issues
    print(
        f"[node] extractor: material={extracted.get('material')} mass_kg={extracted.get('mass_kg')} conf={conf:.2f} issues={issues}"
    )
    return state


__all__ = ["extractor_node"]
