import argparse
import json

from src.download_data import download_dataset
from src.ingest import load_raw_data
from src.validate import validate_raw_data
from src.features import build_feature_table
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict


def main():
    parser = argparse.ArgumentParser(description="Predictive maintenance ML/MLOps CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download")
    sub.add_parser("features")
    sub.add_parser("train")
    sub.add_parser("evaluate")
    sub.add_parser("run-all")

    pred = sub.add_parser("predict")
    pred.add_argument("--machine-id", type=int, required=True)
    pred.add_argument("--timestamp", type=str, required=True)

    args = parser.parse_args()

    if args.command == "download":
        download_dataset()
    elif args.command == "features":
        data = load_raw_data()
        validate_raw_data(data)
        build_feature_table(data)
    elif args.command == "train":
        train_model()
    elif args.command == "evaluate":
        evaluate_model()
    elif args.command == "run-all":
        download_dataset()
        data = load_raw_data()
        validate_raw_data(data)
        features = build_feature_table(data)
        train_model(features)
        evaluate_model()
    elif args.command == "predict":
        print(json.dumps(predict(args.machine_id, args.timestamp), indent=2))


if __name__ == "__main__":
    main()
