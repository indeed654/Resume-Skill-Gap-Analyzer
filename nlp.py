import re
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import docx
import os

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def extract_text_from_pdf(path: str) -> str:
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError("Unsupported file type")

COMMON_SKILLS = [
    "python","java","c++","c#","javascript","react","node.js","node","django",
    "flask","fastapi","sql","postgres","mongodb","docker","kubernetes","aws",
    "azure","gcp","tensorflow","pytorch","pandas","numpy","scikit-learn",
    "nlp","computer vision","linux","git","rest api","graphql","html","css",
    "bash","shell","spark","hadoop","iot","blockchain","smart contract","solidity",
    "docker-compose","ci/cd","terraform","ansible","opencv","matlab","excel"
]

def extract_skills(text: str, skills_list: List[str]=None) -> List[str]:
    if skills_list is None:
        skills_list = COMMON_SKILLS
    text_low = text.lower()
    found = set()
    for s in skills_list:
        s_low = s.lower()
        if re.search(r'\b' + re.escape(s_low) + r'\b', text_low):
            found.add(s)
    return sorted(list(found))

def embed_texts(texts: List[str]):
    model = _load_model()
    embs = model.encode(texts, convert_to_tensor=True)
    return embs

def semantic_similarity(a: str, b: str) -> float:
    model = _load_model()
    emb = model.encode([a, b], convert_to_tensor=True)
    score = util.cos_sim(emb[0], emb[1]).item()
    return float(score * 100)

SKILL_TO_RESOURCES = {
    "python": [
        {"title":"Python for Everybody (Coursera)","url":"https://www.coursera.org/specializations/python"},
        {"title":"Automate the Boring Stuff (book)","url":"https://automatetheboringstuff.com/"}
    ],
    "fastapi":[
        {"title":"FastAPI docs & tutorials","url":"https://fastapi.tiangolo.com/"},
    ],
    "react":[
        {"title":"React Official Tutorial","url":"https://reactjs.org/tutorial/tutorial.html"}
    ],
    "docker":[
        {"title":"Docker Get Started","url":"https://www.docker.com/get-started"}
    ],
    "aws":[
        {"title":"AWS Cloud Practitioner Essentials","url":"https://www.aws.training/"}
    ],
    "nlp":[
        {"title":"Hugging Face Course","url":"https://huggingface.co/learn/nlp-course"}
    ],
    "sql":[
        {"title":"Mode SQL Tutorial","url":"https://mode.com/sql-tutorial/"}
    ],
    "pytorch":[
        {"title":"PyTorch Tutorials","url":"https://pytorch.org/tutorials/"}
    ],
}

def generate_learning_plan(missing_skills: List[str]) -> Dict[str, List[Dict[str,str]]]:
    plan = {}
    for s in missing_skills:
        key = s.lower()
        if key in SKILL_TO_RESOURCES:
            plan[s] = SKILL_TO_RESOURCES[key]
        else:
            plan[s] = [{"title": f"Search resources for {s}","url": f"https://www.google.com/search?q={s}+tutorial"}]
    return plan

def analyze_resume_vs_jd(resume_path: str, jd_text: str) -> Dict[str, Any]:
    resume_text = extract_text_from_file(resume_path)
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    similarity_score = semantic_similarity(resume_text[:2000], jd_text[:2000])
    matched_skills = sorted(list(set(resume_skills).intersection(set(jd_skills))))
    missing_skills = sorted(list(set(jd_skills).difference(set(resume_skills))))
    learning_plan = generate_learning_plan(missing_skills)
    return {
        "match_score": round(similarity_score, 2),
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "learning_plan": learning_plan
    }
