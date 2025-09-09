import re
import unicodedata

from ..state import EstimationState
from ._utils import record_node_trace


def _prompt_yes_no(msg: str) -> bool:
    """Yes/No を標準入力で確認するヘルパー関数。

    振る舞い:
    - プロンプトを表示し、`y`/`yes` は True、`n`/`no` は False を返す。
    - それ以外の入力は再入力を促し、正しい入力があるまでループする。

    Args:
    - msg: プロンプト本文（末尾に " [y/n]: " が付与される）。

    Returns:
    - bool: Yes の場合は True、No の場合は False。
    """
    while True:
        ans = input(f"{msg} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):  # yes
            return True
        if ans in ("n", "no"):  # no
            return False
        print("y か n で回答してください。")


def _parse_mass_kg(s: str) -> float | None:
    """質量(kg)の文字列を数値(float)へ解釈するヘルパー。

    振る舞い:
    - 全角→半角へ正規化し、単位文字（"kg"/"KG"）と桁区切り（","）を除去。
    - 直接 `float(...)` で変換を試み、失敗したら正規表現で最初の数値を抽出して再試行。
    - どちらも失敗した場合は `None` を返す。

    Args:
    - s: ユーザ入力などの生文字列（例: " １,２３kg "、"約 0.85 kg" など）。

    Returns:
    - float | None: kg単位の数値。解釈不能なら `None`。
    """
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
    """HITL（人手確認）で抽出結果を確定させるノード。

    振る舞い:
    - 抽出済みの材料(material)、質量(mass_kg)、信頼度(confidence)を表示
    - ユーザにOK/NGを確認（Yesなら確定し、信頼度を引き上げて次工程へ）
    - NGなら材料/質量の再入力を受け付け、extracted・issues・confidence を更新
    - 入力内容を `human_answers` に保存し、trace に "human_review" を追加

    Args:
    - state: エージェント全体の状態（抽出結果、確信度、HITL回答、エラー、トレース等を含む）。

    Returns:
    - EstimationState: ユーザ確認／修正を反映した新しい状態。
      - `extracted`/`extraction_issues`/`extraction_confidence`/`human_answers`/`needs_human` を更新。
    """

    # 履歴を残す。
    record_node_trace(state, "human_review")

    # 抽出結果を取得
    extracted = dict(state.get("extracted") or {})
    material = extracted.get("material")
    mass_kg = extracted.get("mass_kg")
    conf_overall = state.get("extraction_confidence", 0.0)
    conf_bundle = {}
    try:
        # tools → extractor で渡している詳細信頼度（material/mass_kg/overall）
        conf_bundle = dict(extracted.get("confidence") or {})
    except Exception:
        conf_bundle = {}
    conf_mat = conf_bundle.get("material")
    conf_mass = conf_bundle.get("mass_kg")

    print("[node] human_review: 読み取り結果の確認を行います。")
    print(f"  - 材料(material): {material}")
    print(f"  - 質量(mass_kg): {mass_kg}")
    # 項目別の信頼度を表示（平均値ではなく個別）
    if isinstance(conf_mat, (int, float)):
        print(f"  - 信頼度(material): {float(conf_mat):.2f}")
    if isinstance(conf_mass, (int, float)):
        print(f"  - 信頼度(mass_kg): {float(conf_mass):.2f}")
    # フォールバックとして、項目別が無い場合のみoverallを表示
    if not isinstance(conf_mat, (int, float)) and not isinstance(conf_mass, (int, float)):
        if isinstance(conf_bundle.get("overall"), (int, float)):
            print(f"  - 参考(overall): {float(conf_bundle['overall']):.2f}")
        elif isinstance(conf_overall, (int, float)):
            print(f"  - 参考(overall): {float(conf_overall):.2f}")

    ok = _prompt_yes_no("この読み取り結果で問題ありませんか？")
    if ok:
        state["needs_human"] = False
        # 問題なしの場合は信頼度を少し引き上げる
        # overall を引き上げ（詳細スコアはそのまま）
        base_overall = conf_bundle.get("overall", conf_overall)
        state["extraction_confidence"] = max(float(base_overall if isinstance(base_overall, (int, float)) else 0.0), 0.9)
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

    # issues を更新(未入力項目を追加)
    issues: list[str] = []
    if not extracted.get("material"):
        issues.append("material")
    if extracted.get("mass_kg") is None:
        issues.append("mass_kg")

    # stateを更新して返す
    state["extracted"] = extracted
    state["human_answers"] = {k: extracted.get(k) for k in ("material", "mass_kg")}
    state["needs_human"] = False  # 入力を受けて次工程へ
    state["extraction_issues"] = issues
    # 入力で確度を回復（両方あれば 0.95、片方なら 0.8 程度）
    state["extraction_confidence"] = 0.95 if not issues else (0.8 if len(issues) == 1 else 0.5)
    print(
        "[node] human_review: 入力を反映 "
        f"material={extracted.get('material')} "
        f"mass_kg={extracted.get('mass_kg')} "
        f"issues={issues}"
    )
    return state


__all__ = ["human_in_the_loop_node"]
