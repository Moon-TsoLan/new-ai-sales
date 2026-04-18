from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    product_name: Mapped[str] = mapped_column(String(200), nullable=False)
    product_desc: Mapped[str] = mapped_column(Text, nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    features_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    predicted_sales: Mapped[float] = mapped_column(Float, nullable=False)
    repeat_rate: Mapped[float] = mapped_column(Float, nullable=False)
    average_rating: Mapped[float] = mapped_column(Float, nullable=False)
    shap_plot_path: Mapped[str] = mapped_column(String(500), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="predictions")

