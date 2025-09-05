"""State schema for the Cost Estimation Agent.

Notes
-----
- HITL = Human in the Loop（人間による補正/確認）
- 本エージェントは 2D 図面入力 → 抽出 → 分岐（HITL 有無）→ 原価算出 → 提示 → フィードバック学習
  のフロー（添付図）に対応するフィールドを保持します。
"""

from typing import Any, Dict, List, TypedDict


class EstimationState(TypedDict, total=False):
    input_doc: bytes | str  # 図面ファイル（バイナリ/PDFパスなど）
    meta: Dict[str, Any]  # 顧客/案件メタ
    extracted: Dict[str, Any]  # Extractor の出力（材質/寸法/公差/粗さ/フィーチャ等）
    extraction_confidence: float
    extraction_issues: List[str]  # 抽出の未確定/不足項目
    needs_human: bool  # 人手確認が必要か
    human_answers: Dict[str, Any]  # HITL(Human in the Loop) の回答
    cost_breakdown: Dict[str, Any]  # Calculator の内訳
    total_cost: float | None  # 見積不能時は None を許容
    presentation_payload: Dict[str, Any]  # 提示用データ（根拠リンク等）
    feedback: Dict[str, Any]  # 実績/誤差など
    errors: List[str]
    trace: List[str]  # 通過ノードの記録


__all__ = ["EstimationState"]
