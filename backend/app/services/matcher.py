"""Transparent recursive reference matcher; swap with a trained TRM adapter in production."""
from typing import Any
class RecursiveMatcher:
    def __init__(self, passes: int = 4): self.passes = passes
    def match(self, candidate: dict, job: dict) -> dict[str, Any]:
        required, offered = set(job["skills"]), set(candidate["skills"])
        shared, missing = sorted(required & offered), sorted(required - offered)
        skill = len(shared) / max(1, len(required))
        # Experience is normalized to a stated JD requirement when detectable.
        exp = min(1.0, candidate["years_experience"] / max(1, job["years_experience"])) if job["years_experience"] else .5
        qual = 1.0 if set(candidate["education"]) & set(job["education"]) else .5
        evidence = .60 * skill + .25 * exp + .15 * qual
        state, trace = .0, []
        for step in range(1, self.passes + 1):
            # Shared parameters: same recurrence each pass, not four independent classifiers.
            state = .55 * state + .45 * evidence
            trace.append({"pass": step, "score": round(state * 100, 1), "evidence": round(evidence * 100, 1)})
        return {"score": round(state * 100, 1), "skill_relevance": round(skill*100,1),
          "experience_alignment": round(exp*100,1), "qualification_match": round(qual*100,1),
          "strengths": [f"Matches required skill: {s}" for s in shared] or ["No listed required skills matched"],
          "gaps": [f"Missing requested skill: {s}" for s in missing], "refinement_trace": trace,
          "disclaimer": "Decision support only; recruiter review is required."}
