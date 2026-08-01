from io import BytesIO
from fastapi import HTTPException, UploadFile

def extract_text(upload: UploadFile) -> str:
    data = upload.file.read()
    name = (upload.filename or "").lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(data)).pages)
        if name.endswith(".docx"):
            from docx import Document
            return "\n".join(p.text for p in Document(BytesIO(data)).paragraphs)
        if name.endswith(".txt"):
            return data.decode("utf-8", errors="replace")
    except Exception as exc:
        raise HTTPException(422, f"Could not read {upload.filename}: {exc}")
    raise HTTPException(415, "Only PDF, DOCX, and TXT are supported")
