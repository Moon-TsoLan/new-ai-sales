from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.database import get_db
from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.prediction import PredictionResult
from app.services.file_service import save_user_upload
from app.services.ml_service import predict_and_explain

router = APIRouter(prefix="/predict", tags=["predict"])
settings = get_settings()


def _to_result(item: Prediction) -> PredictionResult:
    model_predictions = item.features_snapshot.get("model_predictions", {}) if isinstance(item.features_snapshot, dict) else {}
    visualizations = item.features_snapshot.get("visualizations", {}) if isinstance(item.features_snapshot, dict) else {}
    return PredictionResult(
        id=item.id,
        product_name=item.product_name,
        product_desc=item.product_desc,
        image_url=f"/uploads/{item.image_path}",
        shap_plot_url=f"/uploads/{item.shap_plot_path}",
        features_snapshot=item.features_snapshot,
        model_predictions=model_predictions,
        visualizations=visualizations,
        predicted_sales=item.predicted_sales,
        repeat_rate=item.repeat_rate,
        average_rating=item.average_rating,
        created_at=item.created_at,
    )


@router.post("", response_model=APIResponse)
def create_prediction(
    product_name: str = Form(...),
    product_desc: str = Form(...),
    image_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    submission_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    rel_image = save_user_upload(user.id, image_file, submission_id=submission_id)
    abs_image = settings.upload_root / Path(rel_image)

    result = predict_and_explain(
        user_id=user.id,
        submission_id=submission_id,
        product_name=product_name,
        product_desc=product_desc,
        uploaded_image_abs=abs_image,
        uploaded_image_rel=rel_image,
    )

    row = Prediction(user_id=user.id, **result)
    db.add(row)
    db.commit()
    db.refresh(row)

    return APIResponse(data=_to_result(row))


@router.get("/{prediction_id}", response_model=APIResponse)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    row = db.query(Prediction).filter(Prediction.id == prediction_id, Prediction.user_id == user.id).first()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    return APIResponse(data=_to_result(row))

