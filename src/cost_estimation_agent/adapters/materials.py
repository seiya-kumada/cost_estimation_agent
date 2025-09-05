import json
import os
from typing import Any, Dict


def _normalize_material_key(s: str) -> str:
    """材料名の照合用にキーを正規化します。

    目的:
    - 材料名の表記ゆれ（全角/半角・大小文字・前後空白）を吸収し、DB検索の安定性を高めます。
    - 型番差異（例: A5052 と A5052-H24）は別物として扱い、同一視しません。

    引数:
    - s: 正規化したい材料名の文字列。

    戻り値:
    - 正規化後の材料キー（NFKC 正規化＋前後空白除去＋大文字化を施した文字列）。
    """
    import unicodedata

    return unicodedata.normalize("NFKC", s).strip().upper()


def _material_prices_file_path() -> str:
    """材料単価JSONファイルの絶対パスを返します。

    説明:
    - 環境変数 `MATERIAL_PRICES_PATH` が設定されていれば、そのパスを優先して返します。
    - 未設定の場合は、既定のパス
      `/home/kumada/projects/cost_estimation_agent/data/material_prices.json`
      を返します。

    引数:
    - なし

    戻り値:
    - `str`: 材料単価JSONファイルの絶対パス。

    Note: Return absolute path to the material prices JSON file.
    """
    override = os.getenv("MATERIAL_PRICES_PATH")
    if override:
        return override
    return "/home/kumada/projects/cost_estimation_agent/data/material_prices.json"


def _load_material_prices_file() -> Dict[str, Any] | None:
    """材料単価のJSONファイルを読み込みます（存在する場合）。

    説明:
    - `_material_prices_file_path()` が返すパスを参照し、ファイルが存在すれば JSON を辞書として読み込んで返します。
    - ファイルが存在しない場合は `None` を返します（呼び出し側で必須扱いにするかは方針次第）。

    引数:
    - なし

    戻り値:
    - `Dict[str, Any]` | `None`: 材料名→単価(JPY/kg)のマッピング。ファイルが無い場合は `None`。

    Note: Load material prices mapping from JSON file if it exists.
    """
    path = _material_prices_file_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _lookup_in_file_mapping(mapping: Dict[str, Any], target_norm: str) -> Dict[str, Any] | None:
    """正規化キーでファイル読み込み済みの材料マップを検索します。

    引数:
    - mapping: 材料名→単価(JPY/kg)の辞書（JSONをロードしたもの）。
    - target_norm: 正規化済みの材料キー（`_normalize_material_key` の出力）。

    戻り値:
    - `dict | None`: 見つかった場合は
        `{found: bool, name: str, unit_price_kg: float|None, source: "file"}`
        を返し、見つからなければ `None` を返します。

    Note: Find a material by normalized key in a file-loaded mapping.
    """
    for material_name, unit_price in mapping.items():
        if _normalize_material_key(str(material_name)) == target_norm:
            try:
                price = float(unit_price)
            except Exception:
                price = None
            return {
                "found": price is not None,
                "name": str(material_name),
                "unit_price_kg": price,
                "source": "file",
            }
    return None


def materials_db_query(name: str | None) -> Dict[str, Any]:
    """材料単価(JPY/kg)をDBから取り出す。

    A5052 と A5052-H24 は別物として扱う（同一視しない）。

    Behavior:
    - ローカル DB ファイルが存在しない場合は FileNotFoundError を送出して処理を中断します。
    - 見つからない材料名は `found=False` を返します。

    Returns:
        {
          "found": bool,
          "name": str | None,            # canonical key in DB
          "unit_price_kg": float | None, # JPY per kg
          "source": "file",
        }
    """
    if not name:
        return {"found": False, "name": None, "unit_price_kg": None, "source": "file"}

    # ローカルDBを必須化：存在しなければ中断
    data_path = _material_prices_file_path()
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Material price DB not found: {data_path}")

    # DB読み込み（失敗時は例外送出）
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            file_db: Dict[str, Any] = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load material price DB at {data_path}: {e}") from e

    # 正規化済みターゲット
    target = _normalize_material_key(name)

    # Lookup in file DB
    found = _lookup_in_file_mapping(file_db, target) if isinstance(file_db, dict) else None
    if found is not None:
        return found

    # 未登録
    return {"found": False, "name": None, "unit_price_kg": None, "source": "file"}


__all__ = ["materials_db_query"]
