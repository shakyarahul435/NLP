# A6 Web Application (React + Django)

This folder contains the assignment web app implemented with minimal libraries and no Bootstrap.

## Backend

Path: `app/backend`

- Framework: Django
- Endpoints:
  - `GET /api/health/`
  - `POST /api/ask/` with JSON body: `{ "question": "..." }`
- Response contains:
  - generated answer
  - source document
  - source chunk id
  - source chunk text

Run backend:

```bash
cd app/backend
python manage.py runserver 127.0.0.1:8000
```

## Frontend

Path: `app/frontend`

- Framework: React (npm + react-scripts)
- Styling: plain CSS (`src/styles.css`)
- No Bootstrap used

Run frontend:

```bash
cd app/frontend
npm install
npm start
```

Frontend calls backend at `http://127.0.0.1:8000/api/ask/`.
