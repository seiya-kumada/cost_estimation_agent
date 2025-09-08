"""External tools and services.

Minimal LLM connection:
- Azure OpenAI を優先して使用（`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_DEPLOYMENT`, 任意で `AZURE_OPENAI_API_VERSION`）。

本ファイルでは以下を提供します:
- GPT-4o を用いた図面画像からの材料・質量抽出（Structured Output）
  - `gpt4o_extract_material_mass(doc)`
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from pydantic import BaseModel

"""
Note: .env の読み込みはエントリポイント（src/main.py）で行います。
このモジュールでは環境変数は直接参照します。
"""


# Materials helpers moved to adapters/materials.py
def materials_db_query(name: str | None) -> Dict[str, Any]:
    """材料名から材料単価情報を取得するファサード関数。

    説明:
    - 実体は `adapters.materials.materials_db_query` に委譲します（疎結合・差し替え容易化）。
    - 呼び出し側は tools を経由することで、内部実装の変更影響を最小化できます。

    Args:
    - name: 材料名（例: "SUS304", "A5052"）。未指定/空は未検出扱い。

    Returns:
    - Dict[str, Any]: 以下のキーを含む辞書。
        - "found": bool — 見つかったか
        - "name": str | None — DB上の正規化名
        - "unit_price_kg": float | None — 単価(JPY/kg)
        - "source": str — 取得元（例: "file"）
    """
    from .adapters.materials import materials_db_query as _impl

    return _impl(name)


def store_history(payload):
    """見積結果の提示ペイロードを保存するためのフック関数（スタブ）。

    説明:
    - 現状は実装されていません（no-op）。永続化の要件に応じて具体化してください。
    - 例: ローカルJSONへの追記、SQLite/外部DBへのINSERT、イベントログ送信など。

    Args:
    - payload: `presentation_node` が生成する提示用ペイロード（辞書）。
        典型的には次のキーを含みます:
        - "summary": {"message": str, "total_cost": float|None}
        - "material_pricing": {"material": str|None, "unit_price_kg": float|None, "mass_kg": float|None}
        - "errors": list[str]

    Returns:
    - None
    """
    # ここで永続化処理を実装してください（例: ファイル/DB/外部API）。
    return None


class MaterialMassOutput(BaseModel):
    """GPT-4o 抽出結果の構造化スキーマ（Pydanticモデル）。

    目的:
    - `gpt4o_extract_material_mass` で使用する応答スキーマを定義します。
    - モデルの Structured Output 機能で、この型に沿った値の生成を促します。

    Attributes:
    - material: 抽出された材料名（例: "SUS304"）。不明な場合は `None`。
    - mass_kg: 抽出・換算済みの質量[kg]。不明な場合は `None`。
    """

    material: Optional[str] = None
    mass_kg: Optional[float] = None
    # Optional evidence snippets (short text from the drawing/OCR supporting each value)
    material_evidence: Optional[str] = None
    mass_kg_evidence: Optional[str] = None


def _to_image_data_url(doc: bytes | str, detail: str = "high") -> Dict[str, Any]:
    """OpenAIの"image_url"コンテンツブロックに変換するヘルパー。

    説明:
    - 入力がファイルパス（str）の場合はバイナリを読み取り、拡張子からMIMEを推定します。
    - 入力がバイト列（bytes）の場合はJPEG相当として扱います。
    - 出力は data URL を含む `{"type": "image_url", "image_url": {"url": ..., "detail": ...}}` 形式。
    - PDFは非対応です（エラー送出）。

    Args:
    - doc: 画像のファイルパス（str）または画像の生バイト列（bytes）。
    - detail: OpenAIの画像詳細レベル（例: "high"）。

    Returns:
    - Dict[str, Any]: OpenAI Chat API（vision）で使用できる image_url ブロック。

    Raises:
    - ValueError: PDFファイルが指定された場合。
    - OSError / IOError: パス指定時のファイル読み込みに失敗した場合。
    """
    if isinstance(doc, str):
        with open(doc, "rb") as f:
            data = f.read()
        ext = (doc.rsplit(".", 1)[-1] if "." in doc else "").lower()
    else:
        data = doc
        ext = "jpeg"

    if ext in {"pdf"}:
        raise ValueError("PDFは対象外です。画像ファイル（png/jpg/jpeg）を指定してください。")

    mime = "image/png" if ext == "png" else "image/jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{b64}",
            "detail": detail,
        },
    }


OCR_READ_VERSION = "v3.2"
OCR_POLL_INTERVAL_S = 0.5
OCR_POLL_MAX_TRIES = 20


def _azure_cv_credentials() -> Optional[tuple[str, str]]:
    """環境変数から Azure CV の資格情報を取得する。揃っていなければ None。"""
    endpoint = os.getenv("AZURE_CV_ENDPOINT")
    key = os.getenv("AZURE_CV_KEY")
    if not (endpoint and key):
        return None
    return endpoint, key


def _load_image_bytes(doc: bytes | str) -> Optional[bytes]:
    """パス/バイト列から画像バイトを取得する。失敗時は None。"""
    if isinstance(doc, str):
        try:
            with open(doc, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"[tools] Azure OCR: 画像読込失敗: {e}")
            return None
    return doc


def _azure_cv_start_analyze(endpoint: str, key: str, payload: bytes) -> Optional[str]:
    """Read Analyze を開始し、Operation-Location を返す。失敗時 None。"""
    analyze_url = endpoint.rstrip("/") + f"/vision/{OCR_READ_VERSION}/read/analyze"
    req = urllib.request.Request(analyze_url, method="POST")
    req.add_header("Ocp-Apim-Subscription-Key", key)
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, data=payload, timeout=15) as resp:
            return resp.headers.get("Operation-Location")
    except urllib.error.HTTPError as e:
        print(f"[tools] Azure OCR analyze失敗: {e}")
        return None
    except Exception as e:
        print(f"[tools] Azure OCR analyze接続失敗: {e}")
        return None


def _azure_cv_poll_result(op_url: str, key: str, interval_s: float, max_tries: int) -> Optional[dict]:
    """Operation-Location をポーリングし、成功時は結果JSONを返す。失敗/timeoutは None。"""
    try:
        for _ in range(max_tries):
            time.sleep(interval_s)
            r = urllib.request.Request(op_url, method="GET")
            r.add_header("Ocp-Apim-Subscription-Key", key)
            with urllib.request.urlopen(r, timeout=15) as resp2:
                data = resp2.read()
            j = json.loads(data.decode("utf-8", errors="ignore"))
            status = str(j.get("status") or "").lower()
            if status == "succeeded":
                return j
            if status == "failed":
                print("[tools] Azure OCR: ステータス failed")
                return None
        print("[tools] Azure OCR: タイムアウト")
        return None
    except Exception as e:
        print(f"[tools] Azure OCR結果取得失敗: {e}")
        return None


def _extract_ocr_lines(payload: dict) -> list[str]:
    """Read結果JSONから行テキストを抽出して返す。"""
    lines: list[str] = []
    ar = payload.get("analyzeResult") or {}
    for page in ar.get("readResults", []) or []:
        for ln in page.get("lines", []) or []:
            t = str(ln.get("text") or "").strip()
            if t:
                lines.append(t)
    return lines


def _normalize_ocr_text(lines: list[str]) -> str:
    """改行・空行を整えたOCRテキストを返す。"""
    raw = "\n".join(lines).strip()
    return "\n".join(s for s in (ln.strip() for ln in raw.splitlines()) if s)


def _azure_ocr_extract_text(doc: bytes | str) -> Optional[str]:
    """Azure Computer Vision Read APIでOCRテキストを抽出する（REST）。"""
    full = _azure_ocr_extract_full(doc)
    return (full or {}).get("text") if full else None


def _azure_ocr_extract_full(doc: bytes | str) -> Optional[Dict[str, Any]]:
    """OCRの全文テキストに加えて、各行のバウンディングボックスも返す。"""
    creds = _azure_cv_credentials()
    if not creds:
        print("[tools] Azure OCR 未設定（AZURE_CV_ENDPOINT/AZURE_CV_KEY）")
        return None
    endpoint, key = creds

    payload = _load_image_bytes(doc)
    if payload is None:
        return None

    op_loc = _azure_cv_start_analyze(endpoint, key, payload)
    if not op_loc:
        print("[tools] Azure OCR: Operation-Location 欠如")
        return None

    result = _azure_cv_poll_result(op_loc, key, OCR_POLL_INTERVAL_S, OCR_POLL_MAX_TRIES)
    if not result:
        return None

    # 文字列本文
    lines_only = _extract_ocr_lines(result)
    text = _normalize_ocr_text(lines_only)

    # ライン+座標
    ocr_lines: list[Dict[str, Any]] = []
    ar = result.get("analyzeResult") or {}
    for page in ar.get("readResults", []) or []:
        pw = page.get("width")
        ph = page.get("height")
        for ln in page.get("lines", []) or []:
            ocr_lines.append(
                {
                    "text": ln.get("text"),
                    "bbox": ln.get("boundingBox"),  # [x1,y1,...,x4,y4]
                    "page_width": pw,
                    "page_height": ph,
                }
            )

    if text:
        print("[tools] Azure OCR: 抽出成功")
        return {"text": text, "lines": ocr_lines}
    return None


def gpt4o_extract_material_mass(doc: bytes | str) -> Dict[str, Any]:
    """図面画像から「材料名」と「質量(kg)」を抽出する（GPT-4o, Structured Output）。

    画像のみの1st passで不足がある場合、OCR（Azure）を併用した2nd passを試行します。
    """
    client, az_deployment, az_detail = _get_azure_openai_client()
    if client is None:
        return {"material": None, "mass_kg": None, "raw": None}

    try:
        image_block = _to_image_data_url(doc, detail=az_detail)  # may raise if file I/O fails
        system = _build_system_prompt()
        user_text = _build_base_user_prompt()

        # 1st pass: image only
        parts1 = _build_user_parts(user_text=user_text, image_block=image_block)
        parsed1 = _chat_parse_material_mass(client, az_deployment, system, parts1)
        if _is_complete(parsed1):
            print("[tools] gpt4o_extract_material_mass: Structured Output (image only)")
            _log_evidence(parsed1)
            # 1st pass 成功時も、可能ならOCRの座標から大まかな位置を推定して表示
            if _ocr_assist_enabled():
                ocr_full_1st = _azure_ocr_extract_full(doc)
                if ocr_full_1st and ocr_full_1st.get("lines"):
                    _log_locations_from_ocr(parsed1, ocr_full_1st["lines"])
            return _to_result(parsed1)

        # 2nd pass: GPT-4o with OCR-assisted text (Azure OCR as source)
        if _ocr_assist_enabled():
            _log_2nd_pass_reasons(parsed1)
            ocr_full = _azure_ocr_extract_full(doc)
            if ocr_full and ocr_full.get("text"):
                parts2 = _build_user_parts_with_ocr(
                    user_text=user_text, image_block=image_block, ocr_text=ocr_full["text"]
                )
                parsed2 = _chat_parse_material_mass(client, az_deployment, system, parts2)
                if parsed2:
                    print("[tools] gpt4o_extract_material_mass: Structured Output (with OCR)")
                    _log_evidence(parsed2)
                    # 位置推定（OCR行の座標から大まかな方位を推測）
                    _log_locations_from_ocr(parsed2, ocr_full.get("lines") or [])
                    return _to_result(parsed2)
            else:
                print("[tools] gpt4o_extract_material_mass: 2nd pass skipped (no OCR text)")

        print("[tools] gpt4o_extract_material_mass: 応答なし/不完全")
        return {"material": None, "mass_kg": None, "raw": None}
    except Exception as e:
        print(f"[tools] GPT-4o抽出に失敗: {e}")
        return {"material": None, "mass_kg": None, "raw": None}


# ===== Internal helpers for GPT-4o extraction =====
def _get_azure_openai_client():
    az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    az_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    az_detail = os.getenv("AZURE_OPENAI_DETAIL", "high")
    if not (az_endpoint and az_key and az_deployment):
        print("[tools] GPT-4o設定不足。material/mass_kgはNoneで返却")
        return None, None, None
    from openai import AzureOpenAI  # type: ignore

    client = AzureOpenAI(api_key=az_key, api_version=az_api_version, azure_endpoint=az_endpoint)
    return client, az_deployment, az_detail


def _build_system_prompt() -> str:
    return (
        "You are an assistant that extracts only 'material' and 'mass_kg' "
        "from a mechanical drawing image. Return structured output using the provided schema."
    )


def _build_base_user_prompt() -> str:
    return (
        "日本語で記載された機械図面画像から、材料名（例: SUS304, A5052, SS400 等）と質量(kg)のみを読み取り、"
        "次のスキーマ（material: string|null, mass_kg: number|null）に従って返してください。"
        "単位がkg以外ならkgへ変換し、小数3桁程度に丸めてください。"
        "可能であれば、各項目を裏付ける短いテキスト断片をそれぞれ"
        "material_evidence と mass_kg_evidence に含めてください（最大50文字程度）。"
    )


def _build_user_parts(*, user_text: str, image_block: Dict[str, Any]) -> list[dict[str, Any]]:
    return [{"type": "text", "text": user_text}, image_block]


def _build_user_parts_with_ocr(*, user_text: str, image_block: Dict[str, Any], ocr_text: str) -> list[dict[str, Any]]:
    max_chars = int(os.getenv("MAX_OCR_CHARS", "4000"))
    snippet = ocr_text[:max_chars]
    assist = (
        "以下はOCRで抽出した文字列です（誤りを含む可能性があります）。必要に応じて参照し、"
        "材料名と質量(kg)のみをスキーマに従って返してください。\nOCR transcript:\n" + snippet
    )
    return [
        {"type": "text", "text": user_text},
        {"type": "text", "text": assist},
        image_block,
    ]


def _chat_parse_material_mass(client, model: str, system: str, parts: list[dict[str, Any]]):
    resp = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": parts},  # type: ignore[arg-type]
        ],
        temperature=0,
        max_tokens=200,
        response_format=MaterialMassOutput,
    )
    return resp.choices[0].message.parsed if resp.choices else None


def _is_complete(parsed) -> bool:
    return bool(parsed and (parsed.material is not None) and (parsed.mass_kg is not None))


def _to_result(parsed) -> Dict[str, Any]:
    return {
        "material": parsed.material,
        "mass_kg": parsed.mass_kg,
        "raw": parsed.model_dump(mode="json"),
        "evidence": {
            "material_evidence": getattr(parsed, "material_evidence", None),
            "mass_kg_evidence": getattr(parsed, "mass_kg_evidence", None),
        },
    }


def _ocr_assist_enabled() -> bool:
    return os.getenv("OCR_ASSIST", "false").lower() in ("1", "true", "yes", "on")


def _log_2nd_pass_reasons(parsed) -> None:
    reasons: list[str] = []
    if not parsed:
        reasons.append("no structured response")
    else:
        if parsed.material is None:
            reasons.append("material missing")
        if parsed.mass_kg is None:
            reasons.append("mass_kg missing")
    print(f"[tools] gpt4o_extract_material_mass: 2nd pass (OCR) trigger reasons: {', '.join(reasons) or 'unknown'}")


def _log_evidence(parsed) -> None:
    try:
        me = getattr(parsed, "material_evidence", None)
        ke = getattr(parsed, "mass_kg_evidence", None)
        if me or ke:
            print("[tools] evidence: material <- " + (me or "(none)"))
            print("[tools] evidence: mass_kg <- " + (ke or "(none)"))
    except Exception:
        pass


def _log_locations_from_ocr(parsed, ocr_lines: list[Dict[str, Any]]) -> None:
    try:
        me = getattr(parsed, "material_evidence", None)
        ke = getattr(parsed, "mass_kg_evidence", None)
        if me:
            desc = _locate_evidence_in_ocr(me, ocr_lines)
            if desc:
                print(f"[tools] location: material ≈ {desc}")
        if ke:
            desc = _locate_evidence_in_ocr(ke, ocr_lines)
            if desc:
                print(f"[tools] location: mass_kg ≈ {desc}")
    except Exception:
        pass


def _locate_evidence_in_ocr(evidence: str, ocr_lines: list[Dict[str, Any]]) -> Optional[str]:
    if not evidence:
        return None
    ev = str(evidence).strip().lower()
    best = None
    best_score = 0
    # 単純な一致/トークン重なりでベスト行を探索
    ev_tokens = [t for t in ev.replace("kg", " kg ").split() if t]
    for ln in ocr_lines:
        text = str(ln.get("text") or "").lower()
        score = 0
        if ev in text:
            score = len(ev)
        else:
            ln_tokens = [t for t in text.replace("kg", " kg ").split() if t]
            overlap = len(set(ev_tokens) & set(ln_tokens))
            score = overlap
        if score > best_score:
            best_score = score
            best = ln
    if not best or not best.get("bbox"):
        return None
    bbox = best["bbox"] or []
    pw = best.get("page_width") or 1
    ph = best.get("page_height") or 1
    try:
        xs = [bbox[i] for i in range(0, len(bbox), 2)]
        ys = [bbox[i] for i in range(1, len(bbox), 2)]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        rx = min(max(cx / pw, 0.0), 1.0)
        ry = min(max(cy / ph, 0.0), 1.0)
        # 方位の粗い分類
        horiz = "左" if rx < 0.33 else ("中央" if rx < 0.66 else "右")
        vert = "上" if ry < 0.33 else ("中央" if ry < 0.66 else "下")
        if horiz == "中央" and vert == "中央":
            pos = "中央付近"
        else:
            pos = f"{horiz}{vert}"
        return f"{pos}（~{int(rx*100)}%, ~{int(ry*100)}%） " f"center=({int(cx)},{int(cy)})px bbox={bbox}"
    except Exception:
        return None


__all__ = [
    "materials_db_query",
    "store_history",
    "gpt4o_extract_material_mass",
]
