"""External tools and services.

Minimal LLM connection:
- Azure OpenAI を優先して使用（`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
  `AZURE_OPENAI_DEPLOYMENT`, 任意で `AZURE_OPENAI_API_VERSION`）。
- 未設定/失敗時はフォールバックの定型質問を返します。

本ファイルでは以下を提供します:
- GPT-4o を用いた図面画像からの材料・質量抽出（Structured Output）
  - `gpt4o_extract_material_mass(doc)`
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

"""
Note: .env の読み込みはエントリポイント（src/main.py）で行います。
このモジュールでは環境変数は直接参照します。
"""

# Materials helpers moved to adapters/materials.py









def ocr_layout_tool(doc): ...


def symbols_gdt_tool(page_images): ...


def geometry_feature_tool(vec_or_mesh): ...


def materials_db_query(name: str | None) -> Dict[str, Any]:
    """Delegates to adapters.materials.materials_db_query."""
    from .adapters.materials import materials_db_query as _impl
    return _impl(name)


def processes_pricing_db_query(req): ...


def machines_db_query(req): ...


def llm_generate_questions(missing_items: List[str]) -> List[str]:
    """HITL向けの確認質問を生成します（Azure OpenAI優先、なければフォールバック）。"""

    # Fallback first (works offline)
    def _fallback(items: List[str]) -> List[str]:
        if not items:
            return [
                "図面の材質（例: SUS304, A5052 等）を指定してください。",
                "表面粗さや公差の規定で特記事項はありますか？",
                "数量と希望納期を教えてください。",
            ]
        return [f"次の不足項目について具体値を教えてください: {it}" for it in items]

    # Azure OpenAI
    az_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    az_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if az_endpoint and az_key and az_deployment:
        try:
            # Prefer the AzureOpenAI client for Azure endpoints
            from openai import AzureOpenAI  # type: ignore

            client = AzureOpenAI(
                api_key=az_key,
                api_version=az_api_version,
                azure_endpoint=az_endpoint,
            )

            items_str = ", ".join(missing_items) if missing_items else "(未指定)"
            sys = (
                "You are a manufacturing cost estimation assistant. Given missing "
                "spec items from a 2D drawing, ask concise Japanese questions to "
                "clarify only what's necessary for estimation. Respond as a JSON "
                "array of strings with 3-6 items, no extra text."
            )
            usr = "不足項目: " + items_str + "\n" "注意: 各質問は1文、具体的・重複なし、単位明記を促すこと。"
            resp = client.chat.completions.create(
                model=az_deployment,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            content = resp.choices[0].message.content if resp.choices else "[]"
            try:
                data = json.loads(content)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    print("[tools] llm_generate_questions: Azure OpenAI(JSON) を使用")
                    return data
            except Exception:
                pass
            lines = [ln.strip("- •* \t") for ln in (content or "").splitlines() if ln.strip()]
            if lines:
                print("[tools] llm_generate_questions: Azure OpenAI(行分割) を使用")
                return lines
        except Exception as e:
            print(f"[tools] Azure OpenAI利用に失敗: {e}. Fallbackに切替")

    # Final fallback
    qs = _fallback(missing_items or [])
    print("[tools] llm_generate_questions: Fallbackを使用")
    return qs


def store_history(payload): ...


# ===== GPT-4o: Structured Output for material & mass =====


class MaterialMassOutput(BaseModel):
    material: Optional[str] = None
    mass_kg: Optional[float] = None


def _to_image_data_url(doc: bytes | str, detail: str = "high") -> Dict[str, Any]:
    """Convert image content to OpenAI image_url content block.

    Accepts a path (str) to an image file or raw bytes. PDFは対象外。
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
    """Use Azure OpenAI GPT-4o to extract material and mass(kg) from an image.

    Returns dict with keys: material, mass_kg, raw
    - material: str | None
    - mass_kg: float | None
    - raw: underlying parsed model as JSON-like dict (or None)
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
        from openai import AzureOpenAI  # type: ignore

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
                {"role": "user", "content": [{"type": "text", "text": user_text}, image_block]},
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
    "ocr_layout_tool",
    "symbols_gdt_tool",
    "geometry_feature_tool",
    "materials_db_query",
    "processes_pricing_db_query",
    "machines_db_query",
    "llm_generate_questions",
    "store_history",
    "gpt4o_extract_material_mass",
]
