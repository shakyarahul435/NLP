# A5 Web App

This app presents the saved A5 outputs and exposes a simple generation API.

It is designed so I can demo results quickly without retraining or re-running the notebook.

## Structure

- `backend/`: Django REST API
- `frontend/`: React UI (npm + react-scripts)
- `docker-compose.yml`: local multi-service run
- `huggingface-space/`: Docker Space files
- `scripts/`: PowerShell scripts for model/Space upload

## Backend API

- `GET /api/health/` - health check
- `GET /api/report/` - metrics, judge table, and loss logs from `a5_outputs/report_assets`
- `POST /api/generate/` - generation from `a5_outputs/dpo_truthful_model`

## Local Run

### 1) Backend

```powershell
cd app/backend
python manage.py migrate
python manage.py runserver
```

### 2) Frontend

```powershell
cd app/frontend
npm install
npm start
```

## Docker Run

```powershell
cd app
docker compose up --build
```

## Notes

- The app reads from `../a5_outputs` and does not retrain the model.
- Uploaded model for this app: `https://huggingface.co/shakyarahul/LLM-as-a-Judge`
- If generation quality looks noisy, check tokenizer/model version compatibility in backend environment.
