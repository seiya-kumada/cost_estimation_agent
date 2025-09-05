Cost Estimation Agent（2D 図面 → 見積）

概要
- 目的: 2D 図面画像から材料・質量を取得し、材料原価を算出してサマリを出力する最小構成のエージェントです。LangGraph を用いてノード型のフローを構築しています。
- フロー: ingestion → extractor → orchestrator →（human_review）→ calculator → presentation
- 対象範囲: 現時点では材料費（円/kg × kg）のみ。将来的に加工費・表面処理・管理費・丸め規則などを拡張しやすい構造です。

主要構成
- グラフ: `src/cost_estimation_agent/graph.py` — LangGraph のビルド。
- ノード群:
  - `src/cost_estimation_agent/nodes/ingestion.py`
  - `src/cost_estimation_agent/nodes/extractor.py` — GPT‑4o（Structured Output）で材料・質量を抽出（任意設定）。
  - `src/cost_estimation_agent/nodes/orchestrator.py` — 信頼度/不足に応じて HITL に分岐。
  - `src/cost_estimation_agent/nodes/human_review.py` — 端末プロンプトで確認・修正を受け付け。
  - `src/cost_estimation_agent/nodes/calculator.py` — 材料費を計算。`make_cost_breakdown` を提供。
  - `src/cost_estimation_agent/nodes/presentation.py` — 結果表示と `presentation_payload` 構築。
- スキーマ: `src/cost_estimation_agent/state.py` — `EstimationState` の定義。
- ツール: `src/cost_estimation_agent/tools.py` — LLM 呼び出しや補助関数、アダプタ。
- 材料DBアダプタ: `src/cost_estimation_agent/adapters/materials.py` — ローカル JSON から単価を読込。

データ
- 既定の単価表: `data/material_prices.json`（編集可）。環境変数 `MATERIAL_PRICES_PATH` で別パスに切替可能。

要件
- Python `>= 3.13`
- 依存関係（PyPI）: `langgraph`, `langchain-core`, `openai`, `pydantic`, `python-dotenv`

セットアップ
- 仮想環境（推奨）: `python -m venv .venv && source .venv/bin/activate`
- 依存インストール（いずれか）:
  - uv（推奨）: `uv sync`
  - pip: `pip install langgraph langchain-core openai pydantic python-dotenv`

設定（.env）
- GPT‑4o（Azure OpenAI、任意）:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT`
  - `AZURE_OPENAI_API_VERSION`（デフォルト: `2024-02-15-preview`）
  - `AZURE_OPENAI_DETAIL`（例: `high`）
- 材料単価ファイルの上書き:
  - `MATERIAL_PRICES_PATH`（JSON への絶対パス）

実行
- CLI: `python -m src.main --image /path/to/drawing.jpg --rfq-id RFQ-001`
- 挙動:
  - Azure OpenAI 未設定時は extractor が `None` を返すため、`human_review` に分岐して手動入力を促す場合があります。
  - calculator は材料DBの `unit_price_kg × mass_kg` を計算し、`state["cost_breakdown"]` を構築します。
  - presentation は結果を表示し、提示用ペイロード（スタブ）を作成します。

補足
- `.env` などのローカルファイルは `.gitignore` で除外しています。
- 費目の追加は `make_cost_breakdown` に拡張を行うと一貫性が保てます。
- しきい値・ロギング・エラー処理は用途に合わせて調整してください。
