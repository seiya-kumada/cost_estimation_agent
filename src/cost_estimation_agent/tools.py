"""外部ツール/サービスとの連携。

提供機能:
- 図面画像からの材料・質量抽出 (GPT‑4o Structured Output): `extract_material_mass_via_gpt4o(doc)`
  - `OCR_ASSIST` が有効かつ `AZURE_CV_ENDPOINT`/`AZURE_CV_KEY` が設定されていれば、
    Azure Computer Vision Read (OCR) と併用した 2nd pass を実行
  - 抽出値に対応する短い根拠テキストを返し、OCR がある場合はおおよその位置(%)に加えて
    中心 px / bbox もログ出力
- 材料単価参照のファサード: `query_materials_db(name)` → `adapters.materials` に委譲
- 見積履歴保存フック: `store_history(payload)`（現在は no-op）

LLM 設定:
- Azure OpenAI を使用（`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`）

補足:
- `.env` の読み込みはエントリポイントで行い、本モジュールは環境変数を直接参照します。
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from pydantic import BaseModel


def query_materials_db(name: str | None) -> Dict[str, Any]:
    """材料名から材料単価情報を取得するファサード関数。

    実体は `adapters.materials.query_materials_db` に委譲します。呼び出し側は tools を
    経由することで、内部実装変更の影響を最小化できます。

    Args:
        name: 材料名（例: "SUS304", "A5052"）。未指定/空は未検出扱い。

    Returns:
        Dict[str, Any]: 次のキーを含む辞書。
            - "found": bool — 見つかったか
            - "name": str | None — DB上の正規化名
            - "unit_price_kg": float | None — 単価(JPY/kg)
            - "source": str — 取得元（例: "file"）
    """
    from .adapters.materials import query_materials_db as _impl

    return _impl(name)


def store_history(payload):
    """見積結果の提示ペイロードを保存するためのフック関数（スタブ）。

    現状は no-op です。永続化要件に応じて、ローカルJSON追記やDB挿入、イベント送信などを
    実装してください。

    Args:
        payload: `presentation_node` が生成する提示用ペイロード（辞書）。
            例: "summary", "material_pricing", "errors" などのキーを含む。

    Returns:
        None
    """
    # ここで永続化処理を実装してください（例: ファイル/DB/外部API）。
    return None


class MaterialMassOutput(BaseModel):
    """GPT-4o 抽出結果の構造化スキーマ（Pydanticモデル）。

    `extract_material_mass_via_gpt4o` の Structured Output で使用する応答スキーマです。

    Attributes:
        material: 抽出された材料名（例: "SUS304"）。不明な場合は `None`。
        mass_kg: 抽出・換算済みの質量[kg]。不明な場合は `None`。
        material_evidence: 材料名の根拠として用いた短いテキスト断片（任意）。
        mass_kg_evidence: 質量の根拠として用いた短いテキスト断片（任意）。
        material_confidence: モデル自己申告の材料に関する信頼度（0.0〜1.0、任意）。
        mass_confidence: モデル自己申告の質量に関する信頼度（0.0〜1.0、任意）。
    """

    material: Optional[str] = None
    mass_kg: Optional[float] = None
    # Optional evidence snippets (short text from the drawing/OCR supporting each value)
    material_evidence: Optional[str] = None
    mass_kg_evidence: Optional[str] = None
    # Model self-reported confidences (0.0–1.0)
    material_confidence: Optional[float] = None
    mass_confidence: Optional[float] = None


def _to_image_data_url(doc: bytes | str, detail: str = "high") -> Dict[str, Any]:
    """OpenAI の `image_url` コンテンツブロックに変換するヘルパー。

    入力がパスの場合はファイルを読み込み、拡張子からMIMEを推定します。入力がバイト列の
    場合は JPEG 相当として扱い、Base64 を含む data URL を生成します。PDF は非対応です。

    Args:
        doc: 画像のファイルパス（str）または画像の生バイト列（bytes）。
        detail: OpenAI の画像詳細レベル（例: "high"）。

    Returns:
        Dict[str, Any]: Chat API（Vision）で使用できる `image_url` ブロック。

    Raises:
        ValueError: PDF ファイルが指定された場合。
        OSError | IOError: パス指定時のファイル読み込みに失敗した場合。
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


def _get_azure_cv_credentials() -> Optional[tuple[str, str]]:
    """`AZURE_CV_ENDPOINT` と `AZURE_CV_KEY` を読み取り、OCR 用資格情報を返す。

    Returns:
        tuple[str, str] | None: `(endpoint, key)`。いずれか未設定の場合は `None`。

    Notes:
        - 呼び出し側は `None` を考慮して分岐してください。
        - 実際の Read API 呼び出しではモジュール定数 `OCR_READ_VERSION`（例: `v3.2`）を使用します。
    """
    endpoint = os.getenv("AZURE_CV_ENDPOINT")
    key = os.getenv("AZURE_CV_KEY")
    if not (endpoint and key):
        return None
    return endpoint, key


def _load_image_bytes(doc: bytes | str) -> Optional[bytes]:
    """パス/バイト列から画像データ（bytes）を取得するユーティリティ。

    引数がファイルパスの場合は `rb` で読み込み、引数がバイト列の場合はそのまま返します。
    読み込み失敗時は `None` を返し、パス指定時は簡易ログを出力します。

    Args:
        doc: 画像ファイルのパス（str）または画像バイト（bytes）。

    Returns:
        bytes | None: 成功時は画像の生バイト。失敗時は `None`。

    Notes:
        - 拡張子や MIME の妥当性検証は行いません（上位で判断）。
        - PDF など非対応の型チェックは呼び出し元（例: `_to_image_data_url`）で実施します。
    """
    if isinstance(doc, str):
        try:
            with open(doc, "rb") as f:
                return f.read()
        except Exception as e:
            print(f"[tools] Azure OCR: 画像読込失敗: {e}")
            return None
    return doc


def _start_cv_read_analysis(endpoint: str, key: str, payload: bytes) -> Optional[str]:
    """Azure Computer Vision Read の分析を開始し、Operation-Location を返す。

    Args:
        endpoint: CV エンドポイント。例 `https://<resource>.cognitiveservices.azure.com`。
        key: Computer Vision (Read) のサブスクリプションキー。
        payload: 送信する画像の生バイト。

    Returns:
        str | None: `Operation-Location` の絶対URL。HTTPエラー/接続失敗/欠如時は `None`。

    Notes:
        - `POST {endpoint}/vision/{OCR_READ_VERSION}/read/analyze` に `application/octet-stream` で送信。
        - 認証は `Ocp-Apim-Subscription-Key: <key>` ヘッダ。
        - タイムアウトは 15 秒。
        - 実際の結果取得は `_poll_cv_read_result` で行います。
    """
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


def _poll_cv_read_result(op_url: str, key: str, interval_s: float, max_tries: int) -> Optional[dict]:
    """Operation-Location をポーリングして Read の結果 JSON を取得する。

    Args:
        op_url: `_start_cv_read_analysis` が返す Operation-Location（analyzeResults の URL）。
        key: Computer Vision (Read) のサブスクリプションキー。
        interval_s: ポーリング間隔（秒）。
        max_tries: 最大試行回数。

    Returns:
        dict | None: 成功時は Read の結果 JSON。失敗・タイムアウト・例外時は `None`。

    Notes:
        - `GET <op_url>` を `Ocp-Apim-Subscription-Key` ヘッダ付きで呼び出し、`status` を判定。
        - `status == "succeeded"` で JSON を返却、`failed` は `None`。
        - 上位の `_extract_full_ocr_result` で本文整形や座標抽出に利用します。
    """
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
    """Read 結果 JSON から行テキストのみを抽出するユーティリティ。

    Azure Computer Vision Read API のレスポンス（`_poll_cv_read_result` の戻り値）を受け取り、
    `analyzeResult.readResults[].lines[].text` をページ・行順で収集します。各行は前後の空白を
    除去し、空文字は除外します。

    Args:
        payload: Read API のレスポンス JSON。

    Returns:
        list[str]: ページ順→行順の行テキスト。座標情報は含みません。

    Notes:
        - ページ幅/高さや boundingBox が必要な場合は `_extract_full_ocr_result` を使用してください。
        - 想定キーが欠けていても例外を投げず、見つからなければ空リストを返します。
    """
    lines: list[str] = []
    ar = payload.get("analyzeResult") or {}
    for page in ar.get("readResults", []) or []:
        for ln in page.get("lines", []) or []:
            t = str(ln.get("text") or "").strip()
            if t:
                lines.append(t)
    return lines


def _normalize_ocr_text(lines: list[str]) -> str:
    """OCR の行テキストを結合・整形して単一の文字列にする。

    各行の前後空白を取り除き、空行を除去したうえで改行で結合します。

    Args:
        lines: OCRで取得した生の行文字列のリスト。

    Returns:
        str: 空行を含まない整形済みの複数行テキスト。
    """
    raw = "\n".join(lines).strip()
    return "\n".join(s for s in (ln.strip() for ln in raw.splitlines()) if s)


def _extract_full_ocr_result(doc: bytes | str) -> Optional[Dict[str, Any]]:
    """Azure OCR の全文テキストと行座標を抽出する。

    画像に対して Azure Computer Vision Read API を実行し、正規化済みの本文テキストと、
    各行のテキスト/バウンディングボックス/ページ寸法を収集して返します。

    Args:
        doc: 画像ファイルのパス（str）または画像バイト（bytes）。

    Returns:
        dict | None: 成功時は以下のキーを持つ辞書。失敗時は `None`。
            - "text": str — 正規化済みの全文テキスト（改行整形済み）
            - "lines": list[dict] — 各行の詳細。各要素は次を含みます。
                {"text": str, "bbox": list[int], "page_width": int|None, "page_height": int|None}

    Notes:
        - 資格情報は `AZURE_CV_ENDPOINT` と `AZURE_CV_KEY` を使用（未設定時は `None` を返します）。
        - 実行フロー: `_load_image_bytes` → `_start_cv_read_analysis` → `_poll_cv_read_result`。
        - `bbox` は [x1, y1, ..., x4, y4] の順でページ座標系の頂点を示します。
    """
    creds = _get_azure_cv_credentials()
    if not creds:
        print("[tools] Azure OCR 未設定（AZURE_CV_ENDPOINT/AZURE_CV_KEY）")
        return None
    endpoint, key = creds

    payload = _load_image_bytes(doc)
    if payload is None:
        return None

    op_loc = _start_cv_read_analysis(endpoint, key, payload)
    if not op_loc:
        print("[tools] Azure OCR: Operation-Location 欠如")
        return None

    result = _poll_cv_read_result(op_loc, key, OCR_POLL_INTERVAL_S, OCR_POLL_MAX_TRIES)
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


def extract_material_mass_via_gpt4o(doc: bytes | str) -> Dict[str, Any]:
    """図面画像から材料名と質量(kg)を抽出する。

    概要:
    - 1st pass: 画像のみで GPT‑4o の Structured Output を取得
    - 2nd pass: 不足時のみ、`OCR_ASSIST` 有効かつ資格情報ありで Azure OCR テキストを併用
    - 可能なら短い根拠テキスト（evidence）を含め、OCRがある場合は概略位置（%）や bbox をログ出力

    Args:
        doc: 画像ファイルのパス（str）または画像バイト（bytes）。

    Returns:
        Dict[str, Any]: 抽出結果の辞書。
            - "material": str | None — 抽出された材料名。
            - "mass_kg": float | None — 抽出・換算済み質量[kg]。
            - "raw": dict | None — モデルの構造化出力の生データ。
            - "evidence": dict | None — 根拠テキスト（"material_evidence", "mass_kg_evidence"）。
            - "confidence": dict | None — 項目別および overall の信頼度（{material, mass_kg, overall}）。

    信頼度の扱い:
        - 1st pass: モデル自己申告の `material_confidence` / `mass_confidence` を採用し、
          利用可能な値の平均を `overall` とします（片方欠落時は取得できた側のみで平均）。
        - 2nd pass: OCR重なり・材料DBヒット・単位チェック等の事後ヒューリスティックで算出し、
          material/mass_kg の平均を `overall` とします。
        - いずれも状況により `confidence` は `None`（または一部フィールド欠落）になり得ます。

    Notes:
        - LLM は Azure OpenAI（`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
          `AZURE_OPENAI_DEPLOYMENT`, 任意で `AZURE_OPENAI_API_VERSION`）で構成します。
        - 2nd pass は `OCR_ASSIST` が true かつ `AZURE_CV_ENDPOINT`/`AZURE_CV_KEY` 設定時のみ。
        - 例外発生や設定不足時はログを出し、material/mass_kg は None を返します。
    """
    cfg = _get_azure_openai_client()
    if cfg is None:
        return {"material": None, "mass_kg": None, "raw": None}
    client, az_deployment, az_detail = cfg

    try:
        # file I/O may raise
        image_block = _to_image_data_url(doc, detail=az_detail)
        system = _build_system_prompt()
        user_text = _build_base_user_prompt()

        # 1st pass: image only
        parsed1 = _run_first_pass(
            client=client,
            model=az_deployment,
            system=system,
            user_text=user_text,
            image_block=image_block,
        )
        if _is_complete(parsed1) and parsed1 is not None:
            print("[tools] extract_material_mass_via_gpt4o: Structured Output (image only)")
            _log_evidence(parsed1)

            # 1st pass の信頼度は LLM 自己申告を採用
            conf_bundle_1st = _build_model_confidence_bundle(parsed1)
            _log_model_first_pass_confidence(conf_bundle_1st)
            return _to_result(parsed1, confidence=conf_bundle_1st)

        # 2nd pass: GPT-4o with OCR-assisted text (Azure OCR as source)
        if _is_ocr_assist_enabled():
            _log_2nd_pass_reasons(parsed1)
            ocr_full = _extract_full_ocr_result(doc)
            if ocr_full and ocr_full.get("text"):
                parsed2 = _run_second_pass(
                    client=client,
                    model=az_deployment,
                    system=system,
                    user_text=user_text,
                    image_block=image_block,
                    ocr_text=ocr_full["text"],
                )
                if parsed2:
                    print("[tools] extract_material_mass_via_gpt4o: Structured Output (with OCR)")
                    _log_evidence(parsed2)
                    # 位置推定（OCR行の座標から大まかな方位を推測）
                    ocr_lines_2nd = ocr_full.get("lines") or []
                    _log_locations_from_ocr(parsed2, ocr_lines_2nd)
                    _log_second_pass_confidence(parsed2, ocr_lines_2nd)
                    conf_bundle_2nd = _compute_confidence_bundle(parsed2, ocr_lines_2nd)
                    return _to_result(parsed2, confidence=conf_bundle_2nd)
            else:
                print("[tools] extract_material_mass_via_gpt4o: 2nd pass skipped (no OCR text)")

        print("[tools] extract_material_mass_via_gpt4o: 応答なし/不完全")
        return {"material": None, "mass_kg": None, "raw": None}
    except Exception as e:
        print(f"[tools] GPT-4o抽出に失敗: {e}")
        return {"material": None, "mass_kg": None, "raw": None}


def _get_azure_openai_client() -> Optional[tuple[Any, str, str]]:
    """Azure OpenAIクライアントと設定値を取得する。

    環境変数からAzure OpenAIを初期化し、クライアント、デプロイメント名、
    画像 `detail` 設定を返します。必須の環境変数が不足している場合は `None` を返します。

    Returns:
        tuple[Any, str, str] | None: `(client, deployment, detail)`。
            - `client`: `openai.AzureOpenAI` のインスタンス
            - `deployment`: Azure OpenAI のデプロイメント名（モデル指定）
            - `detail`: 画像詳細レベル（例: "high"）

    Notes:
        - 使用環境変数: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
          `AZURE_OPENAI_DEPLOYMENT`, 任意で `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DETAIL`。
    """
    az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    az_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    az_detail = os.getenv("AZURE_OPENAI_DETAIL", "high")
    if not (az_endpoint and az_key and az_deployment):
        print("[tools] GPT-4o設定不足。material/mass_kgはNoneで返却")
        return None
    from openai import AzureOpenAI  # type: ignore

    client = AzureOpenAI(api_key=az_key, api_version=az_api_version, azure_endpoint=az_endpoint)
    return client, az_deployment, az_detail


def _build_system_prompt() -> str:
    """LLM 用のシステムプロンプトを構築して返す。

    図面画像から抽出すべき項目（material, mass_kg）を明示し、
    返却形式が構造化出力であることを指示する短い定常文です。

    Returns:
        str: システムロールに渡す英語のプロンプト文字列。
    """
    return (
        "You are an assistant that extracts only 'material' and 'mass_kg' "
        "from a mechanical drawing image. Return structured output using the provided schema."
    )


def _build_base_user_prompt() -> str:
    """ユーザープロンプト（日本語）の定型文を構築して返す。

    図面画像から抽出すべき項目（材料名と質量[kg]）、出力スキーマ、単位換算、
    可能であれば根拠テキスト(evidence)も含める旨を日本語で明示します。
    OCR 併用時は `_build_user_parts_with_ocr` 側でこの文面にOCR断片が追記されます。

    Returns:
        str: ユーザー発話として送る日本語プロンプト文字列。
    """
    return (
        "日本語で記載された機械図面画像から、材料名（例: SUS304, A5052, SS400 等）と質量(kg)のみを読み取り、"
        "次のスキーマ（material: string|null, mass_kg: number|null）に従って返してください。"
        "単位がkg以外ならkgへ変換し、小数3桁程度に丸めてください。"
        "可能であれば、各項目を裏付ける短いテキスト断片をそれぞれ"
        "material_evidence と mass_kg_evidence に含めてください（最大50文字程度）。"
        "さらに、各値の確からしさを 0.0〜1.0 の実数で material_confidence / mass_confidence に記入してください。"
        "「材質」「材料」などの語句の近傍にある文字列は材料名である確率が高いです。"
        "「質量」「重量」「kg」「g」などの語句の近傍にある語句は質量である確率が高いです。"
        "1.0=明確に図面上で根拠が確認できる、0.0=断定不能 とし、中間は根拠の明確さや曖昧さに応じて一貫して採点してください。"
    )


def _build_user_parts(*, user_text: str, image_block: Dict[str, Any]) -> list[dict[str, Any]]:
    """Vision入力用のユーザー発話コンテンツを構築する。

    Chat Completions（Vision）に渡す `content` 配列の一部を作り、
    テキスト指示と画像ブロック（`image_url`）を並べて返します。

    Args:
        user_text: 日本語の指示文（抽出対象・スキーマなど）。
        image_block: `_to_image_data_url` が返す `{"type":"image_url", ...}` ブロック。

    Returns:
        list[dict[str, Any]]: `[{"type":"text", ...}, image_block]` の配列。
    """
    return [{"type": "text", "text": user_text}, image_block]


def _build_user_parts_with_ocr(*, user_text: str, image_block: Dict[str, Any], ocr_text: str) -> list[dict[str, Any]]:
    """OCRテキストを併用するユーザー発話コンテンツを構築する。

    画像に対する日本語の指示文に加え、OCRで抽出した文字列のスニペットを
    追加したうえで、Vision入力用の `content` 配列を返します。

    Args:
        user_text: 日本語の指示文（抽出対象・スキーマなど）。
        image_block: `_to_image_data_url` が返す `{"type":"image_url", ...}` ブロック。
        ocr_text: Azure OCR から得た全文テキスト。

    Returns:
        list[dict[str, Any]]: `[{"type":"text", ...}, {"type":"text", ...}, image_block]` の配列。

    Notes:
        - `MAX_OCR_CHARS`（既定 4000）で `ocr_text` を先頭から切り詰めて使用します。
        - OCRは誤りを含む旨を明記し、参照補助としてのみ用いることを促します。
    """
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


def _run_first_pass(
    *, client, model: str, system: str, user_text: str, image_block: Dict[str, Any]
) -> Optional[MaterialMassOutput]:
    """画像のみで Structured Output を得る 1st pass を実行する。

    GPT‑4o に対して、システム/ユーザー指示と画像ブロックのみを与え、
    材料名と質量(kg)の構造化出力を取得します（OCRは併用しません）。

    Args:
        client: Azure OpenAI のクライアントインスタンス。
        model: 使用するデプロイメント名。
        system: システムプロンプト文字列。
        user_text: ユーザープロンプト文字列。
        image_block: `_to_image_data_url` が返す `image_url` ブロック。

    Returns:
        MaterialMassOutput | None: 構造化応答のパース結果。失敗時は `None`。
    """
    parts = _build_user_parts(user_text=user_text, image_block=image_block)
    return _parse_material_mass_via_chat(client, model, system, parts)


def _run_second_pass(
    *,
    client,
    model: str,
    system: str,
    user_text: str,
    image_block: Dict[str, Any],
    ocr_text: str,
) -> Optional[MaterialMassOutput]:
    """OCR テキストを併用して Structured Output を得る 2nd pass を実行する。

    1st pass で十分な情報が得られない場合に、OCR で抽出したテキストを
    ユーザー発話の補助として追加し、GPT‑4o の構造化応答を取得します。

    Args:
        client: Azure OpenAI のクライアントインスタンス。
        model: 使用するデプロイメント名。
        system: システムプロンプト文字列。
        user_text: ユーザープロンプト文字列。
        image_block: `_to_image_data_url` が返す `image_url` ブロック。
        ocr_text: Azure OCR から得た全文テキスト。

    Returns:
        MaterialMassOutput | None: 構造化応答のパース結果。失敗時は `None`。

    Notes:
        - 実際のメッセージ構築は `_build_user_parts_with_ocr` に委譲します。
        - `MAX_OCR_CHARS` で OCR テキストは先頭から切り詰められます。
    """
    parts = _build_user_parts_with_ocr(user_text=user_text, image_block=image_block, ocr_text=ocr_text)
    return _parse_material_mass_via_chat(client, model, system, parts)


def _parse_material_mass_via_chat(client, model: str, system: str, parts: list[dict[str, Any]]):
    """Chat Completions(parse) を用いて構造化応答を取得する。

    Azure OpenAI の Chat Completions の `parse` 機能を利用し、
    `MaterialMassOutput`（Pydantic）へ直接パースされた結果を返します。

    Args:
        client: Azure OpenAI のクライアントインスタンス。
        model: 使用するデプロイメント名。
        system: システムプロンプト文字列。
        parts: ユーザーメッセージの `content` 配列（テキスト/画像/補助テキストなど）。

    Returns:
        MaterialMassOutput | None: 構造化応答のパース結果。選択肢が空の場合は `None`。

    Notes:
        - `response_format=MaterialMassOutput`, `temperature=0`, `max_tokens=200` を指定しています。
        - SDK 側の例外は上位で捕捉してください（本関数では握りません）。
    """
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


def _is_complete(parsed: Optional[MaterialMassOutput]) -> bool:
    """抽出結果が完了要件を満たすかを判定する。

    完了要件は `material` と `mass_kg` の両方が `None` でないこと。

    Args:
        parsed: 構造化応答のパース結果。`None` の可能性あり。

    Returns:
        bool: 両フィールドが存在して値を持つ場合に `True`。
    """
    return bool(parsed and (parsed.material is not None) and (parsed.mass_kg is not None))


def _to_result(parsed: MaterialMassOutput, *, confidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """構造化応答を最終返却用の辞書に整形する。

    `MaterialMassOutput`（Pydantic）から、呼び出し側へ返す辞書を生成します。
    返却値は抽出値に加えて、モデルの生データ（`raw`）と `evidence` を含みます。
    `evidence` フィールド自体は常に含めますが、各値は未取得の場合に `None` になり得ます。
    `confidence` は指定されたときのみ含めます。

    Args:
        parsed: 構造化応答の Pydantic モデル。
        confidence: 項目別の信頼度バンドル（例: {"material": float|None, "mass_kg": float|None, "overall": float|None}）。
            未指定の場合は返却辞書に含めません。

    Returns:
        Dict[str, Any]: 次のキーを持つ辞書。
            - `material`: str | None — 抽出された材料名。
            - `mass_kg`: float | None — 抽出・換算済みの質量[kg]。
            - `raw`: dict — `parsed.model_dump(mode="json")` によるモデルの生データ。
            - `evidence`: dict — `material_evidence` / `mass_kg_evidence` を格納（値は None 可）。
            - `confidence`: dict | 省略 — 指定時のみ追加（material/mass_kg/overall を想定）。
    """
    out = {
        "material": parsed.material,
        "mass_kg": parsed.mass_kg,
        "raw": parsed.model_dump(mode="json"),
        "evidence": {
            "material_evidence": getattr(parsed, "material_evidence", None),
            "mass_kg_evidence": getattr(parsed, "mass_kg_evidence", None),
        },
    }
    if confidence is not None:
        out["confidence"] = confidence
    return out


def _is_ocr_assist_enabled() -> bool:
    """OCR 併用（2nd pass）を有効にする設定かを判定する。

    環境変数 `OCR_ASSIST` の値が有効トークン（`1`, `true`, `yes`, `on` のいずれか・大文字小文字無視）
    に一致する場合に `True` を返します。

    Returns:
        bool: OCR 併用が有効なら `True`、それ以外は `False`。
    """
    return os.getenv("OCR_ASSIST", "false").lower() in ("1", "true", "yes", "on")


def _log_2nd_pass_reasons(parsed: Optional[MaterialMassOutput]) -> None:
    """2nd pass（OCR併用）を試みる理由をログ出力する。

    1st pass の構造化応答を確認し、`material`/`mass_kg` の欠落や応答自体の
    欠如を根拠として、2nd pass のトリガー理由を標準出力へ記録します。

    Args:
        parsed: 1st pass の構造化応答。`None` の場合は「no structured response」を記録。

    Returns:
        None
    """
    reasons: list[str] = []
    if not parsed:
        reasons.append("no structured response")
    else:
        if parsed.material is None:
            reasons.append("material missing")
        if parsed.mass_kg is None:
            reasons.append("mass_kg missing")
    print(
        f"[tools] extract_material_mass_via_gpt4o: 2nd pass (OCR) trigger reasons: {', '.join(reasons) or 'unknown'}"
    )


def _log_evidence(parsed: MaterialMassOutput) -> None:
    """抽出結果に含まれる根拠テキストをログ出力する。

    Args:
        parsed: 構造化応答のPydanticモデル。`material_evidence`／`mass_kg_evidence` を
            含む可能性がある。

    Returns:
        None
    """
    try:
        me = getattr(parsed, "material_evidence", None)
        ke = getattr(parsed, "mass_kg_evidence", None)
        if me or ke:
            print("[tools] evidence: material <- " + (me or "(none)"))
            print("[tools] evidence: mass_kg <- " + (ke or "(none)"))
    except Exception:
        pass


def _log_locations_from_ocr(parsed: MaterialMassOutput, ocr_lines: list[Dict[str, Any]]) -> None:
    """OCR行座標を用いて evidence のおおよその位置をログ出力する。

    `parsed` に含まれる `material_evidence` / `mass_kg_evidence` を、与えられた
    `ocr_lines`（各行の `text`, `bbox`, `page_width`, `page_height` を含む辞書のリスト）
    と照合し、最も合致する行の相対位置（％）や中心px/bboxを推定して出力します。

    Args:
        parsed: 構造化応答（evidence を含む可能性あり）。
        ocr_lines: Azure OCR から得た行情報のリスト。

    Returns:
        None
    """
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
    """OCR行から evidence に最も合致する行を見つけ位置記述を返す。

    単純一致またはトークン重なりによるスコアリングで最適な行を選び、
    その行の `bbox` とページ幅・高さから相対位置（％）を算出します。返り値は
    「方位（%）, center=(x,y)px bbox=[...]」形式の短い文字列です。

    Args:
        evidence: 図面からの根拠テキスト断片。
        ocr_lines: Azure OCR の行情報のリスト（`text`, `bbox`, `page_width`, `page_height`）。

    Returns:
        str | None: 推定位置の記述。該当行が見つからない、または座標がない場合は `None`。
    """
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
        return f"{pos}（~{int(rx * 100)}%, ~{int(ry * 100)}%） " f"center=({int(cx)},{int(cy)})px bbox={bbox}"
    except Exception:
        return None


def _evaluate_ocr_overlap(evidence: Optional[str], ocr_lines: Optional[list[Dict[str, Any]]]) -> float:
    """OCR行とevidenceの重なり度合いを0.0–1.0で評価する。

    evidence をトークン化して各OCR行と比較し、最大の重なり率（overlap/len(evidence_tokens)）
    を返します。OCRが無い、または evidence が無い場合は 0.0 を返します。

    Args:
        evidence: 抽出根拠テキスト。
        ocr_lines: Azure OCR の行情報リスト。

    Returns:
        float: 0.0〜1.0 の重なりスコア。
    """
    if not evidence or not ocr_lines:
        return 0.0
    ev = str(evidence).strip().lower()
    if not ev:
        return 0.0
    ev_tokens = [t for t in ev.replace("kg", " kg ").split() if t]
    if not ev_tokens:
        return 0.0
    best = 0.0
    for ln in ocr_lines:
        text = str((ln or {}).get("text") or "").lower()
        if not text:
            continue
        if ev in text:
            # 文字列として含まれる場合は高スコア（上限1.0）
            best = max(best, 1.0)
            continue
        ln_tokens = [t for t in text.replace("kg", " kg ").split() if t]
        if not ln_tokens:
            continue
        overlap = len(set(ev_tokens) & set(ln_tokens))
        best = max(best, overlap / max(len(ev_tokens), 1))
        if best >= 1.0:
            break
    return float(max(0.0, min(1.0, best)))


def _check_material_in_db(name: Optional[str]) -> bool:
    """材料名が材料単価DBに存在するかを判定する。

    adapters.materials の `query_materials_db` を用いて、与えられた材料名が
    単価DBに登録されているかを真偽で返します。`name` が空/None の場合や、
    参照中に例外が発生した場合は `False` を返します。

    Args:
        name: 照会する材料名（例: "SUS304", "A5052"）。空/None は未指定とみなす。

    Returns:
        bool: DBで見つかれば `True`、それ以外は `False`。

    Notes:
        - 実体は `query_materials_db` に委譲します。照合は内部で正規化されるため、
          大文字小文字・全角半角の表記ゆれにある程度耐性があります。
        - 例外は握りつぶして `False` を返し、上位のスコアリングでの加点のみを抑制します。
    """
    if not name:
        return False
    try:
        res = query_materials_db(name)
        return bool(res.get("found"))
    except Exception:
        return False


def _estimate_material_confidence(
    parsed: MaterialMassOutput, ocr_lines: Optional[list[Dict[str, Any]]]
) -> tuple[float, list[str]]:
    """材料の信頼度を事後推定する（OCR行を任意で利用）。

    スコアリングルール（上限1.0、最終的に0.0〜1.0へクリップ）:
    - +0.50: `material` の値が存在する
    - +0.20: `material_evidence` が存在する
    - +0.20: 材料DBヒット（`query_materials_db(material)` が found）
    - +0.00〜+0.20: OCR重なり — `material_evidence` と OCR 行の重なり率 ov∈[0,1] に比例（0.2×ov）

    根拠は短い文字列の配列 `basis` として返し、例えば
    `["value", "evidence", "db_hit", "ocr_overlap=0.73"]` の形式になります。

    Args:
        parsed: 構造化応答（`material` と `material_evidence` を参照）。
        ocr_lines: Azure OCR の行情報（`text`/`bbox`/`page_width`/`page_height` を含む辞書の配列）
            または `None`（未使用）。

    Returns:
        tuple[float, list[str]]: `(score, basis)` のタプル。

    Notes:
        - 1st pass では LLM の自己申告信頼度を採用しており、本関数は主に 2nd pass の評価に用います。
        - OCR が無い場合は重なり加点は 0 になり、他の加点のみでスコアが決まります。
    """
    score = 0.0
    basis: list[str] = []
    mat = getattr(parsed, "material", None)
    ev = getattr(parsed, "material_evidence", None)

    if mat:
        score += 0.5
        basis.append("value")
    if ev:
        score += 0.2
        basis.append("evidence")
    if _check_material_in_db(mat):
        score += 0.2
        basis.append("db_hit")
    ov = _evaluate_ocr_overlap(ev, ocr_lines)
    if ov > 0:
        # 最大+0.2（重なりに比例）
        add = 0.2 * ov
        score += add
        basis.append(f"ocr_overlap={ov:.2f}")
    score = float(max(0.0, min(1.0, score)))
    return score, basis


def _estimate_mass_confidence(
    parsed: MaterialMassOutput, ocr_lines: Optional[list[Dict[str, Any]]]
) -> tuple[float, list[str]]:
    """質量(kg)の信頼度を事後推定する（OCR行を任意で利用）。

    スコアリングルール（上限1.0、最終的に0.0〜1.0へクリップ）:
    - +0.50: `mass_kg` の値が存在する
    - +0.20: `mass_kg_evidence` に「kg/ｋｇ」が含まれる（単位明示）
    - +0.10: `mass_kg` が正の数（妥当性の簡易チェック）
    - +0.00〜+0.20: OCR重なり — `mass_kg_evidence` と OCR 行の重なり率 ov∈[0,1] に比例（0.2×ov）

    根拠は短い文字列の配列 `basis` として返し、例えば
    `["value", "unit_kg", "plausible", "ocr_overlap=0.78"]` の形式になります。

    Args:
        parsed: 構造化応答（`mass_kg` と `mass_kg_evidence` を参照）。
        ocr_lines: Azure OCR の行情報（`text`/`bbox`/`page_width`/`page_height` を含む辞書の配列）
            または `None`（未使用）。

    Returns:
        tuple[float, list[str]]: `(score, basis)` のタプル。

    Notes:
        - 1st pass では LLM の自己申告信頼度を採用しており、本関数は主に 2nd pass の評価に用います。
        - OCR が無い場合は重なり加点は 0 になり、他の加点のみでスコアが決まります。
    """
    score = 0.0
    basis: list[str] = []
    mass = getattr(parsed, "mass_kg", None)
    ev = getattr(parsed, "mass_kg_evidence", None)

    if mass is not None:
        score += 0.5
        basis.append("value")
    ev_l = str(ev or "").lower()
    if ev and ("kg" in ev_l or "ｋｇ" in ev_l):
        score += 0.2
        basis.append("unit_kg")
    if isinstance(mass, (int, float)) and mass > 0:
        score += 0.1
        basis.append("plausible")
    ov = _evaluate_ocr_overlap(ev, ocr_lines)
    if ov > 0:
        add = 0.2 * ov
        score += add
        basis.append(f"ocr_overlap={ov:.2f}")
    score = float(max(0.0, min(1.0, score)))
    return score, basis


def _compute_confidence_bundle(
    parsed: MaterialMassOutput, ocr_lines: Optional[list[Dict[str, Any]]]
) -> Dict[str, Any]:
    """事後推定ヒューリスティックで信頼度を集計し、バンドル化する。

    材料と質量それぞれの信頼度を `_estimate_material_confidence`／`_estimate_mass_confidence`
    で算出し、その平均を `overall` として返します。OCR行 `ocr_lines` は任意で、
    与えられた場合は evidence とOCRの重なりに基づく加点が反映されます。

    Args:
        parsed: 構造化応答（`material`/`mass_kg` と各 evidence を参照）。
        ocr_lines: Azure OCR の行情報（`text`/`bbox`/`page_width`/`page_height`）の配列、または `None`。

    Returns:
        Dict[str, Any]: 次のキーを持つ信頼度バンドル。
            - `material`: float — 材料の信頼度（0.0–1.0）
            - `mass_kg`: float — 質量の信頼度（0.0–1.0）
            - `overall`: float — 上記2つの平均（0.0–1.0）

    Notes:
        - 主に 2nd pass（OCR併用）の評価で使用します。
        - 1st pass（画像のみ）ではモデル自己申告の `_build_model_confidence_bundle` を使用します。
    """
    mat_score, _ = _estimate_material_confidence(parsed, ocr_lines)
    mass_score, _ = _estimate_mass_confidence(parsed, ocr_lines)
    overall = float((mat_score + mass_score) / 2.0)
    return {
        "material": mat_score,
        "mass_kg": mass_score,
        "overall": overall,
    }


def _build_model_confidence_bundle(parsed: MaterialMassOutput) -> Dict[str, Any]:
    """モデル自己申告の信頼度からバンドルを構築する（1st pass 用）。

    `MaterialMassOutput` が持つ `material_confidence` と `mass_confidence` をそのまま用い、
    利用可能な要素の平均を `overall` として計算します（片方のみでも平均に採用）。
    いずれの値も無い場合は `overall` は `None` となります。ヒューリスティック加点や
    クリップ処理は行いません。

    Args:
        parsed: モデルの構造化応答（自己申告信頼度フィールドを含み得る）。

    Returns:
        Dict[str, Any]: `{"material": float|None, "mass_kg": float|None, "overall": float|None}`。
    """
    mat = getattr(parsed, "material_confidence", None)
    mass = getattr(parsed, "mass_confidence", None)
    # overall は取得できた要素の平均（片方のみでも採用する）。
    vals: list[float] = []
    if isinstance(mat, (int, float)):
        vals.append(float(mat))
    if isinstance(mass, (int, float)):
        vals.append(float(mass))
    overall: Optional[float] = (sum(vals) / len(vals)) if vals else None
    out: Dict[str, Any] = {
        "material": float(mat) if isinstance(mat, (int, float)) else None,
        "mass_kg": float(mass) if isinstance(mass, (int, float)) else None,
        "overall": overall,
    }
    return out


def _log_model_first_pass_confidence(conf: Dict[str, Any]) -> None:
    """モデル自己申告の 1st pass 信頼度を整形してログ出力する。

    入力 `conf` は `_build_model_confidence_bundle` が返す辞書を想定し、
    `material` と `mass_kg` をそれぞれ小数2桁に整形して出力します。
    値が存在しない場合は `(none)` と表示します。ログ用途のため、
    内部で例外が発生しても握りつぶして処理を継続します。

    Args:
        conf: `{"material": float|None, "mass_kg": float|None, "overall": float|None}` を想定する辞書。

    Returns:
        None
    """
    try:
        m = conf.get("material")
        k = conf.get("mass_kg")

        def fmt(x):
            return f"{float(x):.2f}" if isinstance(x, (int, float)) else "(none)"

        print("[tools] confidence(1st-model): " f"material={fmt(m)} mass_kg={fmt(k)}")
    except Exception:
        pass


def _log_second_pass_confidence(parsed: MaterialMassOutput, ocr_lines: Optional[list[Dict[str, Any]]]) -> None:
    """2nd pass の項目別信頼度を整形してログ出力する。

    `parsed`（2nd pass の構造化応答）から、事後推定ヒューリスティック
   （`_estimate_material_confidence`／`_estimate_mass_confidence`）で material と
    mass_kg のスコアを算出し、それぞれの basis（根拠内訳）とともに 1 行で出力します。
    `ocr_lines` が与えられていれば、OCR 重なりに基づく加点もスコアに反映されます。

    Args:
        parsed: 2nd pass の構造化応答（evidence を含む可能性あり）。
        ocr_lines: Azure OCR の行情報（`text`/`bbox`/`page_width`/`page_height`）配列、または `None`。

    Returns:
        None

    Notes:
        - ログ用途のため、内部例外は握りつぶして処理を継続します。
    """
    try:
        mat_score, mat_basis = _estimate_material_confidence(parsed, ocr_lines)
        mass_score, mass_basis = _estimate_mass_confidence(parsed, ocr_lines)
        print(
            "[tools] confidence(2nd): "
            f"material={mat_score:.2f} basis=[{', '.join(mat_basis)}] "
            f"mass_kg={mass_score:.2f} basis=[{', '.join(mass_basis)}]"
        )
    except Exception:
        pass


__all__ = [
    "query_materials_db",
    "store_history",
    "extract_material_mass_via_gpt4o",
]
