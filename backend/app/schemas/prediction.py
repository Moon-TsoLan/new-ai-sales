from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PredictionResult(BaseModel):
    id: int
    product_name: str
    product_desc: str
    image_url: str
    shap_plot_url: str
    features_snapshot: dict[str, Any]
    model_predictions: dict[str, Any] | None = None
    visualizations: dict[str, Any] | None = None
    predicted_sales: float
    repeat_rate: float
    average_rating: float
    created_at: datetime


class PredictionListItem(BaseModel):
    id: int
    product_name: str
    predicted_sales: float
    created_at: datetime


class HistoryPage(BaseModel):
    items: list[PredictionListItem]
    total: int
    page: int
    page_size: int

