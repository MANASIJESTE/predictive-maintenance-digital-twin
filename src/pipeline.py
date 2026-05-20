from src.download_data import download_dataset
from src.ingest import load_raw_data
from src.validate import validate_raw_data
from src.features import build_feature_table
from src.train import train_model
from src.evaluate import evaluate_model


def run_all():
    download_dataset()
    data = load_raw_data()
    validate_raw_data(data)
    features = build_feature_table(data)
    train_model(features)
    evaluate_model()


if __name__ == "__main__":
    run_all()
