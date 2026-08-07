from mlops.train_production import main as train_main
from mlops.promote_if_better import main as promote_main


def main() -> None:
    print("=== Weekly training ===")
    train_main()

    print("=== Conditional promotion ===")
    promote_main()

    print("=== Weekly train-and-promote completed ===")


if __name__ == "__main__":
    main()