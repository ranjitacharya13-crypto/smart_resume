# SmartHire AI

CPU-friendly, explainable resume screening with a JAX-powered recursive matching model. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow, deployment, and training plan.

## Run locally

### Windows (PowerShell)

Open two PowerShell windows from the project folder.

```powershell
# Terminal 1: API
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```powershell
# Terminal 2: dashboard
cd frontend
npm.cmd install
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
npm.cmd run dev
```

Open `http://localhost:3000`. `npm.cmd` is intentional: it avoids the
PowerShell execution-policy error raised by `npm.ps1` on some Windows systems.

### macOS / Linux

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
