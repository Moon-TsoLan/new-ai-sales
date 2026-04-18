from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Product Sales Predictor"
    api_prefix: str = "/api"

    database_url: str = "sqlite:///./app.db"
    secret_key: str = "change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_minutes: int = 60 * 24 * 7

    project_root: Path = Path(__file__).resolve().parents[3]
    upload_root: Path = project_root / "uploads"

    # Existing model pipeline paths
    bert_base_path: Path = project_root / "model" / "bert-base-uncased"
    bert_classifier_path: Path = project_root / "model" / "bert_classifier" / "best_classifier.pth"
    bert_category_encoder_path: Path = project_root / "model" / "bert_classifier" / "category_encoder.pkl"
    bert_sub_encoder_path: Path = project_root / "model" / "bert_classifier" / "sub_encoder.pkl"
    bert_ner_path: Path = project_root / "model" / "bert_ner_agent" / "bert_ner_agent.pth"
    style_kmeans_path: Path = project_root / "model" / "resnet50" / "A_style_kmeans.pkl"
    regression_model_path: Path = project_root / "model" / "regression_model" / "best_regression_model.pth"
    tree_model_path: Path = project_root / "model" / "decision_tree_model" / "best_decision_tree_model.pth"
    background_csv: Path = project_root / "data" / "processed" / "A_final_input.csv"

    max_upload_size_mb: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Avoid creating multiple sqlite files due to different startup cwd.
    if settings.database_url.strip() == "sqlite:///./app.db":
        db_file = (settings.project_root / "backend" / "app.db").resolve().as_posix()
        settings.database_url = f"sqlite:///{db_file}"
    settings.upload_root.mkdir(parents=True, exist_ok=True)
    return settings

