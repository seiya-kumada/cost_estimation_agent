import re
import unicodedata

from ..state import EstimationState


def _prompt_yes_no(msg: str) -> bool:
    while True:
        ans = input(f"{msg} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):  # yes
            return True
        if ans in ("n", "no"):  # no
            return False
        print("y か n で回答してください。")


def _parse_mass_kg(s: str) -> float | None:
    # 全角→半角、単位/区切り除去してから数値化
    t = unicodedata.normalize("NFKC", s).strip()
    t = t.replace("kg", "").replace("KG", "")
    t = t.replace(",", "")
    t = t.strip()
    try:
        return float(t)
    except Exception:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None


def human_in_the_loop_node(state: EstimationState) -> EstimationState:
    # 取得結果と確信度を提示 → ユーザが確認（Yes/No）→ No の場合は再入力を受け付けて更新
    trace = state.get("trace", [])
    trace.append("human_review")
    state["trace"] = trace

    extracted = dict(state.get("extracted") or {})
    material = extracted.get("material")
    mass_kg = extracted.get("mass_kg")
    conf = state.get("extraction_confidence", 0.0)

    print("[node] human_review: 読み取り結果の確認を行います。")
    print(f"  - 材料(material): {material}")
    print(f"  - 質量(mass_kg): {mass_kg}")
    print(f"  - 信頼度(confidence): {conf:.2f}")

    ok = _prompt_yes_no("この読み取り結果で問題ありませんか？")
    if ok:
        state["needs_human"] = False
        # 問題なしの場合は信頼度を少し引き上げる
        state["extraction_confidence"] = max(conf, 0.9)
        state["extraction_issues"] = []
        print("[node] human_review: ユーザ確認でOK。次工程へ進みます。")
        return state

    # ユーザ修正フロー
    print("[node] human_review: 修正入力をお願いします。未指定は空Enterで保持します。")
    new_material = input("材料（例: SUS304, A5052, SS400）: ").strip()
    if new_material:
        extracted["material"] = new_material

    while True:
        raw = input("質量(kg)を数値で入力（例: 1.23）: ").strip()
        if not raw:
            break
        parsed = _parse_mass_kg(raw)
        if parsed is not None:
            extracted["mass_kg"] = parsed
            break
        print("数値として解釈できませんでした。再入力してください。")

    # issues を更新
    issues: list[str] = []
    if not extracted.get("material"):
        issues.append("material")
    if extracted.get("mass_kg") is None:
        issues.append("mass_kg")

    state["extracted"] = extracted
    state["human_answers"] = {k: extracted.get(k) for k in ("material", "mass_kg")}
    state["needs_human"] = False  # 入力を受けて次工程へ
    state["extraction_issues"] = issues
    # 入力で確度を回復（両方あれば 0.95、片方なら 0.8 程度）
    state["extraction_confidence"] = 0.95 if not issues else (0.8 if len(issues) == 1 else 0.5)
    print(
        f"[node] human_review: 入力を反映 material={extracted.get('material')} mass_kg={extracted.get('mass_kg')} issues={issues}"
    )
    return state


__all__ = ["human_in_the_loop_node"]
