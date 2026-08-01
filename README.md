# SmartHire AI

CPU-friendly, explainable resume screening with a JAX-powered recursive matching model. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow, deployment, and training plan.

## Run locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# In another terminal
cd ../frontend && npm install && npm run dev
```

The API uses SQLite by default for a zero-setup demo. Set `DATABASE_URL=postgresql+psycopg://...` for Supabase/Postgres. The dashboard expects `NEXT_PUBLIC_API_URL=http://localhost:8000`.

> Screening results are decision support, not an automated hiring decision. Recruiters should review evidence, comply with applicable employment law, and not use protected characteristics.
