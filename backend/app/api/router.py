from fastapi import APIRouter

from app.api.routes.auth import router as auth_router
from app.api.routes.history import router as history_router
from app.api.routes.predict import router as predict_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(predict_router)
api_router.include_router(history_router)

