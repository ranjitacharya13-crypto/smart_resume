from typing import Annotated
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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

@app.get("/health")
def health(): return {"status":"ok", "matcher":"recursive-reference-v1", "cpu_only":True}

@app.post("/jobs")
def create_job(title: Annotated[str, Form()], description: Annotated[str, Form()] = "", file: UploadFile | None = File(None), db: Session = Depends(get_db)):
    text = description.strip() or (extract_text(file) if file else "")
    if not text: raise HTTPException(422, "Provide a description or a JD document")
    row = Job(title=title, description=text, parsed=parse_document(text)); db.add(row); db.commit(); db.refresh(row)
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
