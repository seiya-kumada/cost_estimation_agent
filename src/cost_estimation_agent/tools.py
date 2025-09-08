"""External tools and services.

Minimal LLM connection:
- Azure OpenAI を優先して使用（`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_DEPLOYMENT`, 任意で `AZURE_OPENAI_API_VERSION`）。

本ファイルでは以下を提供します:
- GPT-4o を用いた図面画像からの材料・質量抽出（Structured Output）
  - `gpt4o_extract_material_mass(doc)`
"""

import base64
import os
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


# ===== GPT-4o: Structured Output for material & mass =====


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


def gpt4o_extract_material_mass(doc: bytes | str) -> Dict[str, Any]:
    """図面画像から「材料名」と「質量(kg)」を抽出する（GPT-4o, Structured Output）。

    説明:
    - 入力の画像（バイト列またはファイルパス）を data URL 化し、Azure OpenAI GPT-4o に提示。
    - Pydantic モデル `MaterialMassOutput` を `response_format` として指定し、構造化出力で
      `material` と `mass_kg` を安全にパースします。
    - Azure の接続情報が未設定の場合や失敗時は、両項目とも `None` を返します（ログ出力あり）。

    必要な環境変数:
    - `AZURE_OPENAI_ENDPOINT`: Azure OpenAI エンドポイント URL
    - `AZURE_OPENAI_API_KEY`: API キー
    - `AZURE_OPENAI_DEPLOYMENT`: デプロイ名（モデル指定）
    - `AZURE_OPENAI_API_VERSION`（任意, 既定: "2024-02-15-preview"）
    - `AZURE_OPENAI_DETAIL`（任意, 既定: "high"）

    Args:
    - doc: 画像のバイト列（bytes）またはファイルパス（str）。PDF は非対応。

    Returns:
    - Dict[str, Any]: 以下のキーを含む辞書。
        - "material": str | None — 抽出された材料名（例: "SUS304"）
        - "mass_kg": float | None — 抽出・換算済みの質量[kg]
        - "raw": dict | None — モデルの構造化応答を JSON 風辞書で保持

    Notes:
    - 内部で例外は捕捉し、`{"material": None, "mass_kg": None, "raw": None}` を返却します。
    - 入力がファイルパスの場合の読み込みや PDF 指定は `_to_image_data_url` に委譲します。
    """
    az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    az_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    az_detail = os.getenv("AZURE_OPENAI_DETAIL", "high")

    if not (az_endpoint and az_key and az_deployment):
        print("[tools] GPT-4o設定不足。material/mass_kgはNoneで返却")
        return {"material": None, "mass_kg": None, "raw": None}

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=az_key,
            api_version=az_api_version,
            azure_endpoint=az_endpoint,
        )

        image_block = _to_image_data_url(doc, detail=az_detail)
        system = (
            "You are an assistant that extracts only 'material' and 'mass_kg' "
            "from a mechanical drawing image. Return structured output using the provided schema."
        )
        user_text = (
            "日本語で記載された機械図面画像から、材料名（例: SUS304, A5052, SS400 等）と質量(kg)のみを読み取り、"
            "次のスキーマ（material: string|null, mass_kg: number|null）に従って返してください。"
            "単位がkg以外ならkgへ変換し、小数3桁程度に丸めてください。余計な説明や補足は不要です。"
        )

        resp = client.beta.chat.completions.parse(
            model=az_deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": user_text}, image_block]},  # type: ignore
            ],
            temperature=0,
            max_tokens=200,
            response_format=MaterialMassOutput,
        )

        parsed = resp.choices[0].message.parsed if resp.choices else None
        if parsed:
            print("[tools] gpt4o_extract_material_mass: Structured Output を使用")
            return {
                "material": parsed.material,
                "mass_kg": parsed.mass_kg,
                "raw": parsed.model_dump(mode="json"),
            }
        print("[tools] gpt4o_extract_material_mass: 応答なし")
        return {"material": None, "mass_kg": None, "raw": None}
    except Exception as e:
        print(f"[tools] GPT-4o抽出に失敗: {e}")
        return {"material": None, "mass_kg": None, "raw": None}


__all__ = [
    "materials_db_query",
    "store_history",
    "gpt4o_extract_material_mass",
]
