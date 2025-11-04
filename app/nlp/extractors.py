# app/nlp/extractors.py
import re
from io import BytesIO
from typing import List
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from app.nlp.normalizer import normalize_skills

# Whitelisted skills/hints (lowercase)
_SKILL_HINTS = {
    "python","pandas","numpy","scikit-learn","sklearn","pytorch","tensorflow",
    "sql","mysql","postgres","postgresql","spark","airflow","dbt",
    "aws","azure","gcp","docker","kubernetes","linux","git","bash",
    "javascript","typescript","node.js","react","java","c++","scala",
    "ml","machine learning","nlp","llm","fastapi","flask","s3","dbt","pyspark",
    "yolov5","pytorch","opencv","keras","hadoop","redshift","snowflake","bigquery"
}

_DEGREE_PAT = re.compile(r"\b(B\.?Tech|B\.?E\.?|BSc|BS|MSc|MS|M\.?Tech|MBA|PhD)\b", re.I)
_YEARS_PAT  = re.compile(r"(\b\d{1,2})\+?\s*(?:years|yrs)\s*(?:of)?\s*(?:experience|exp)?", re.I)
_UNI_PAT    = re.compile(r"\b(University of [A-Za-z][A-Za-z\s]+|[A-Za-z][A-Za-z\s]+ University|UTA|UT Arlington|Andhra University)\b", re.I)

_EMAIL_PAT  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_PAT  = re.compile(r"(\+?\d[\d\-\s\(\)]{7,}\d)")
_HANDLE_PAT = re.compile(r"\b@[A-Za-z0-9_]+\b")
_URL_PAT    = re.compile(r"https?://\S+")
_NUMISH_PAT = re.compile(r"^(?:\d+|\d+[kmb%]?|\d{2,4}(?:\.\d+)?[a-z%]?)$", re.I)  # 100k, 3.7, 2025, 99%

def _extract_text_from_pdf(data: bytes) -> str:
    with BytesIO(data) as bio:
        return pdf_extract_text(bio) or ""

def _extract_text_from_docx(data: bytes) -> str:
    with BytesIO(data) as bio:
        doc = Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)

def sniff_and_extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return _extract_text_from_docx(data)
    try:
        return data.decode("utf-8", "ignore")
    except Exception:
        return ""

def _scrub_pii(text: str) -> str:
    text = _EMAIL_PAT.sub(" [email] ", text)
    text = _PHONE_PAT.sub(" [phone] ", text)
    text = _HANDLE_PAT.sub(" [handle] ", text)
    text = _URL_PAT.sub(" [url] ", text)
    return text

def _tokenize_candidates(text: str) -> List[str]:
    # split on non-alphanum while preserving tech tokens with dots/plus/hash
    raw = re.split(r"[^A-Za-z0-9\.\+\#\-]+", text.lower())
    norm = []
    for t in raw:
        if not t:
            continue
        t = t.replace("scikit-learn", "sklearn")
        if t in {"nodejs","node"}:
            t = "node.js"
        norm.append(t)
    return norm

def _keep_token(t: str) -> bool:
    if not t or len(t) < 2:
        return False
    # drop obvious numbers/years/percents/amounts
    if _NUMISH_PAT.match(t):
        return False
    # keep known skills OR tokens with tech punctuation (c++, node.js)
    if t in _SKILL_HINTS:
        return True
    if any(sym in t for sym in [".", "+", "#"]):
        # allow things like c++, node.js
        return True
    return False

def simple_skill_tokens(text: str) -> List[str]:
    text = _scrub_pii(text)
    tokens = _tokenize_candidates(text)
    return [t for t in tokens if _keep_token(t)]

def extract_resume_entities(text: str) -> dict:
    text_scrub = _scrub_pii(text)
    degrees = list(set(_DEGREE_PAT.findall(text_scrub) or []))
    uni = list(set(_UNI_PAT.findall(text_scrub) or []))
    yrs = 0.0
    m = _YEARS_PAT.search(text_scrub)
    if m:
        try:
            yrs = float(m.group(1))
        except:
            pass

    raw_skills = simple_skill_tokens(text_scrub)
    skills = normalize_skills(raw_skills)

    return {
        "skills": skills,
        "edu": [{"degree": d} for d in degrees],
        "orgs": uni,
        "exp_years": yrs
    }
