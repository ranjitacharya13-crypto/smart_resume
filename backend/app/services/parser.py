import re
SKILLS = {"python","java","javascript","typescript","react","next.js","tailwind","tailwind css","fastapi","flask","sql","postgresql","aws","docker","kubernetes","pytorch","machine learning","nlp","spacy","excel","tableau","git","linux","agile","tensorflow"}
EDUCATION = {"phd","doctorate","master","mba","bachelor","bsc","msc","degree","certification"}
REQUIRED_MARKERS = ("required", "must have", "must-have", "essential", "mandatory", "minimum qualification")
PREFERRED_MARKERS = ("preferred", "nice to have", "nice-to-have", "bonus", "a plus", "desired")
ROLE_TERM_STOP_WORDS = {"experience", "years", "year", "knowledge", "understanding", "ability", "skills", "skill", "candidate", "work", "working", "strong", "good", "excellent", "team", "environment", "development", "developer", "engineer", "and", "or", "with", "in", "of", "the", "to", "a"}

def extract_role_terms(text: str) -> list[str]:
    """Extract explicit job terms without requiring them to be in SKILLS.

    This deliberately uses only phrases placed after requirement/preference
    labels, keeping the result explainable and avoiding broad keyword guessing.
    """
    terms = set()
    for section in re.split(r"[\n.!;]+", text.lower()):
        if any(marker in section for marker in PREFERRED_MARKERS):
            continue
        marker = re.search(r"(?:required|must[- ]have|essential|mandatory|preferred|nice[- ]to[- ]have|bonus|desired)\s*:?[\s]*", section)
        if not marker:
            continue
        remainder = re.sub(r"\b\d{1,2}\+?\s*(?:years?|yrs?)\b", "", section[marker.end():])
        for item in re.split(r",|/|\band\b|\bor\b", remainder):
            term = re.sub(r"[^a-z0-9.+# -]", "", item).strip(" -")
            words = term.split()
            if not term or len(words) > 3 or all(word in ROLE_TERM_STOP_WORDS for word in words):
                continue
            if not any(char.isalpha() for char in term):
                continue
            terms.add(term)
    return sorted(terms)

def classify_job_skills(text: str, skills: list[str]) -> tuple[list[str], list[str]]:
    """Separate explicit must-haves from preferences in a job description."""
    lower = text.lower()
    required, preferred = set(), set()
    sections = [section for section in re.split(r"[\n.!;]+", lower) if section.strip()]
    for skill in skills:
        relevant_sections = [section for section in sections if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", section)]
        if any(marker in section for section in relevant_sections for marker in REQUIRED_MARKERS):
            required.add(skill)
        elif any(marker in section for section in relevant_sections for marker in PREFERRED_MARKERS):
            preferred.add(skill)

    # A short, unstructured job description normally lists only requirements.
    if not required:
        required = set(skills) - preferred
    preferred -= required
    return sorted(required), sorted(preferred)

def parse_document(text: str) -> dict:
    lower = text.lower()
    skills = sorted(s for s in SKILLS if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", lower))
    years = [int(x) for x in re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)", lower)]
    education = sorted(e for e in EDUCATION if re.search(r"(?<!\w)"+e+r"(?!\w)", lower))
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    required_skills, preferred_skills = classify_job_skills(text, skills)
    return {"skills": skills, "required_skills": required_skills, "preferred_skills": preferred_skills,
            "role_terms": extract_role_terms(text),
            "years_experience": max(years, default=0), "education": education,
            "work_history": lines[:20], "text": text[:12000]}
