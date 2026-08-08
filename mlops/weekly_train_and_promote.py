from mlops.train_production import main as train_main
from mlops.promote_if_better import main as promote_main
from modeling.predict_gauge_24h_production import main as predict_main


def main() -> None:
    print("=== Weekly candidate training ===")
    train_main()

    print("=== Validation model selection ===")
    winning_model_version = promote_main()
    print(f"Selected model: {winning_model_version}")

    print("=== Prediction with selected production model ===")
    predict_main()

    print("=== Weekly train, select, and predict completed ===")


if __name__ == "__main__":
    main()