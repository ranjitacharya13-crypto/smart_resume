from typing import Annotated
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session
from .database import Base, engine, get_db
from .models import Candidate, Job, MatchScore
from .services.extraction import extract_text
from .services.parser import parse_document
from .services.matcher import RecursiveMatcher
Base.metadata.create_all(bind=engine)
app = FastAPI(title="SmartHire AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])
matcher = RecursiveMatcher()

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    return """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>SmartHire AI API</title>
    <style>body{font:16px system-ui;margin:8vh auto;max-width:42rem;padding:2rem;color:#173c33;background:#f4f5ef}a{color:#c6532d}</style></head>
    <body><h1>SmartHire AI</h1><p>The API is running.</p><p>Open the <a href=\"http://localhost:3000\">dashboard</a> or view the <a href=\"/docs\">API documentation</a>.</p></body></html>"""

@app.get("/favicon.ico", include_in_schema=False, status_code=204)
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health(): return {"status":"ok", "matcher":"recursive-reference-v1", "cpu_only":True}

@app.post("/jobs")
def create_job(title: Annotated[str, Form()], description: Annotated[str, Form()] = "", file: UploadFile | None = File(None), db: Session = Depends(get_db)):
    text = description.strip() or (extract_text(file) if file else "")
    if not text: raise HTTPException(422, "Provide a description or a JD document")
    # The role title often contains important role context (for example,
    # "Backend Engineer") that is absent from a short job description.
    row = Job(title=title, description=text, parsed=parse_document(f"{title}\n{text}")); db.add(row); db.commit(); db.refresh(row)
    return {"id":row.id, "title":row.title, "parsed":row.parsed}

@app.post("/jobs/{job_id}/resumes")
def upload_resumes(job_id: int, files: Annotated[list[UploadFile], File()], db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job: raise HTTPException(404, "Job not found")
    created=[]
    for upload in files:
        text = extract_text(upload)
        parsed = parse_document(text); name = (upload.filename or "Candidate").rsplit('.',1)[0].replace('_',' ')
        c = Candidate(name=name, source_filename=upload.filename or '', raw_text=text, parsed=parsed); db.add(c); db.flush()
        explanation = matcher.match(parsed, job.parsed)
        db.add(MatchScore(job_id=job.id, candidate_id=c.id, score=explanation["score"], explanation=explanation))
        created.append({"candidate_id":c.id,"name":name,"score":explanation["score"]})
    db.commit(); return {"uploaded":created}

@app.get("/jobs/{job_id}/shortlist")
def shortlist(job_id: int, db: Session = Depends(get_db)):
    job=db.get(Job, job_id)
    if not job: raise HTTPException(404,"Job not found")
    rows=db.query(MatchScore, Candidate).join(Candidate, Candidate.id==MatchScore.candidate_id).filter(MatchScore.job_id==job_id).order_by(MatchScore.score.desc()).all()
    return {"job":{"id":job.id,"title":job.title}, "candidates":[{"id":c.id,"name":c.name,"filename":c.source_filename,"score":m.score, **m.explanation} for m,c in rows]}
