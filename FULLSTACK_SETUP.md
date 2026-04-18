# Fullstack Setup

## Backend

1. `cd backend`
2. `pip install -r requirements.txt`
3. `uvicorn app.main:app --reload --port 8000`

## Frontend

1. `cd frontend`
2. `npm install`
3. `npm run dev`

## API

- Register: `POST /api/auth/register`
- Login: `POST /api/auth/login`
- Refresh: `POST /api/auth/refresh`
- Predict: `POST /api/predict` (`multipart/form-data`)
- History: `GET /api/history`
- History detail: `GET /api/history/{id}`
- Delete history: `DELETE /api/history/{id}`

