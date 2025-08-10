from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.nlp import analyze_resume_vs_jd
from app.models import AnalysisResponse
import uvicorn
import tempfile
import shutil
import os

app = FastAPI(title="Resume Skill Gap Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(resume: UploadFile = File(...), jd_text: str = Form(...)):
    # save uploaded file to a temp location
    suffix = os.path.splitext(resume.filename)[1].lower()
    if suffix not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail="Only .pdf, .docx, .txt allowed for resume")

    tmp_dir = tempfile.mkdtemp()
    try:
        resume_path = os.path.join(tmp_dir, resume.filename)
        with open(resume_path, "wb") as f:
            shutil.copyfileobj(resume.file, f)

        result = analyze_resume_vs_jd(resume_path, jd_text)
        return result
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
