from dotenv import load_dotenv
from cost_estimation_agent.graph import compile_app
from cost_estimation_agent.state import EstimationState
import argparse
import os


def main() -> None:
    # .env をエントリポイントで読み込む（環境変数の統一管理）
    load_dotenv(override=True)
    app = compile_app(checkpointer=None)  # 実装では checkpointer を設定推奨

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        help="入力図面の画像ファイルパス（png/jpg/jpeg）",
        default="/home/kumada/data/cost_estimation_agent/inputs/samples/sample_00.jpg",
    )
    parser.add_argument(
        "--rfq-id",
        help="RFQ ID",
        default="RFQ-001",
    )
    args = parser.parse_args()

    input_doc: bytes | str
    if args.image and os.path.exists(args.image):
        input_doc = args.image  # 画像パスをそのまま渡す（GPT-4oが参照）
    else:
        print(f"[main] 画像パスが無効です: {args.image}. ダミー入力で続行します。")
        input_doc = b"dummy-image"

    initial_state: EstimationState = {
        "input_doc": input_doc,
        "meta": {"rfq_id": args.rfq_id, "input_type": "2D-image"},
    }

    # 1回分の推論（疑似）
    result_state = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "rfq-001"}},
    )
    # 通過ノードとサマリを表示
    trace = result_state.get("trace", [])
    print(f"[trace] path={' -> '.join(trace)}")
    print(f"[result] total_cost={result_state.get('total_cost')} breakdown={result_state.get('cost_breakdown')}")

    # 実装では:
    # - human_review は外部UIで回答取得→再投入（.update() / .invoke()）する運用


if __name__ == "__main__":
    main()
