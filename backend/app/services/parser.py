import re
SKILLS = {"python","java","javascript","typescript","react","next.js","fastapi","flask","sql","postgresql","aws","docker","kubernetes","pytorch","machine learning","nlp","spacy","excel","tableau","git","linux","agile","tensorflow"}
EDUCATION = {"phd","doctorate","master","mba","bachelor","bsc","msc","degree","certification"}
def parse_document(text: str) -> dict:
    lower = text.lower()
    skills = sorted(s for s in SKILLS if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", lower))
    years = [int(x) for x in re.findall(r"(\d{1,2})\+?\s*(?:years?|yrs?)", lower)]
    education = sorted(e for e in EDUCATION if re.search(r"(?<!\w)"+e+r"(?!\w)", lower))
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return {"skills": skills, "years_experience": max(years, default=0), "education": education,
            "work_history": lines[:20], "text": text[:12000]}
