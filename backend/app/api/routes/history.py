from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.prediction import HistoryPage, PredictionListItem
from app.services.file_service import remove_relative_dir

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=APIResponse)
def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    query = db.query(Prediction).filter(Prediction.user_id == user.id)
    total = query.count()
    rows = (
        query.order_by(Prediction.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    items = [
        PredictionListItem(
            id=x.id,
            product_name=x.product_name,
            predicted_sales=x.predicted_sales,
            created_at=x.created_at,
        )
        for x in rows
    ]
    return APIResponse(data=HistoryPage(items=items, total=total, page=page, page_size=page_size))


@router.get("/{prediction_id}", response_model=APIResponse)
def get_history_detail(
    prediction_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    row = db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.user_id == user.id).first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History item not found.")
    return APIResponse(
        data={
            "id": row.id,
            "product_name": row.product_name,
            "product_desc": row.product_desc,
            "image_url": f"/uploads/{row.image_path}",
            "shap_plot_url": f"/uploads/{row.shap_plot_path}",
            "features_snapshot": row.features_snapshot,
            "model_predictions": row.features_snapshot.get("model_predictions", {})
            if isinstance(row.features_snapshot, dict)
            else {},
            "visualizations": row.features_snapshot.get("visualizations", {})
            if isinstance(row.features_snapshot, dict)
            else {},
            "predicted_sales": row.predicted_sales,
            "repeat_rate": row.repeat_rate,
            "average_rating": row.average_rating,
            "created_at": row.created_at,
        }
    )


@router.delete("/{prediction_id}", response_model=APIResponse)
def delete_history(
    prediction_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    row = db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.user_id == user.id).first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="History item not found.")

    submission_dir = Path(row.image_path).parent
    if str(submission_dir) not in {".", ""}:
        remove_relative_dir(str(submission_dir).replace("\\", "/"))
    db.delete(row)
    db.commit()
    return APIResponse(data=True)

