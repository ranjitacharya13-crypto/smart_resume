"""JAX CPU reference implementation of SmartHire's recursive matcher.

This deliberately small, interpretable model uses a single recurrence across all
passes. A trained ~7M-parameter JAX/Flax TRM can replace `recursive_scores`
without changing this class's output contract.
"""
from typing import Any
from functools import partial
import re
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("passes",))
def recursive_scores(evidence: jax.Array, passes: int = 4) -> jax.Array:
    """Apply the same state update repeatedly and return every state.

    `evidence` is [must-have coverage, preferred-skill coverage, experience,
    qualification]. Each pass rechecks the same job-relevant evidence.
    The weights make each factor explicit; in a trained TRM they would be learned.
    """
    weighted_evidence = jnp.dot(evidence, jnp.array([0.55, 0.10, 0.25, 0.10]))
    def refine(state: jax.Array, _: jax.Array):
        next_state = 0.55 * state + 0.45 * weighted_evidence
        return next_state, next_state
    _, states = jax.lax.scan(refine, jnp.array(0.0), xs=None, length=passes)
    return states

class RecursiveMatcher:
    def __init__(self, passes: int = 4):
        if passes < 1: raise ValueError("passes must be at least 1")
        self.passes = passes

    def match(self, candidate: dict, job: dict) -> dict[str, Any]:
        # Recruiters first assess stated must-haves, then preferences. Only
        # job-related evidence is considered; protected characteristics are not.
        required = set(job.get("required_skills") or job.get("role_terms") or job.get("skills", []))
        preferred = set(job.get("preferred_skills", [])) - required
        resume_text = candidate.get("text", "").lower()
        dynamic_matches = {term for term in required | preferred
                           if term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", resume_text)}
        offered = set(candidate.get("skills", [])) | dynamic_matches
        shared, missing = sorted(required & offered), sorted(required - offered)
        preferred_shared = sorted(preferred & offered)
        preferred_missing = sorted(preferred - offered)
        skill = len(shared) / max(1, len(required))
        preferred_skill = len(preferred_shared) / max(1, len(preferred)) if preferred else 1.0

        required_years = job.get("years_experience", 0)
        candidate_years = candidate.get("years_experience", 0)
        exp = min(1.0, candidate_years / required_years) if required_years else .7
        required_education = set(job.get("education", []))
        qual = 1.0 if not required_education else (1.0 if required_education & set(candidate.get("education", [])) else .35)

        evidence = jnp.array([skill, preferred_skill, exp, qual], dtype=jnp.float32)
        states = recursive_scores(evidence, self.passes).tolist()
        weighted = float(jnp.dot(evidence, jnp.array([.55, .10, .25, .10])))
        convergence = 1 - (0.55 ** self.passes)
        trace = [{"pass": step, "score": round(float(state) / convergence * 100, 1),
                  "evidence": round(weighted * 100, 1)} for step, state in enumerate(states, 1)]
        # Missing most must-haves is a recruiter-review signal, not an automatic rejection.
        review_cap = .49 if required and skill < .5 else 1.0
        score = round(min(weighted, review_cap) * 100, 1)
        recommendation = ("Strong shortlist" if skill >= .8 and exp >= .8 else
                          "Recruiter review required" if skill >= .5 else
                          "Does not currently meet most stated must-haves")
        strengths = ([f"Meets must-have: {s}" for s in shared] +
                     [f"Matches preferred skill: {s}" for s in preferred_shared])
        if candidate_years and required_years and candidate_years >= required_years:
            strengths.append(f"Meets stated experience level ({candidate_years} years)")
        gaps = [f"Missing requested skill: {s}" for s in missing]
        gaps += [f"Nice-to-have not evidenced: {s}" for s in preferred_missing]
        if required_years and candidate_years < required_years:
            gaps.append(f"Experience evidenced: {candidate_years} years; role requests {required_years}")
        breakdown = [
            {"label": "Must-have skills", "weight": 55, "evidence": round(skill * 100, 1),
             "points": round(skill * 55, 1),
             "detail": f"{len(shared)} of {len(required)} stated must-haves evidenced" if required else "No structured must-have skills were detected in this job brief"},
            {"label": "Preferred skills", "weight": 10, "evidence": round(preferred_skill * 100, 1),
             "points": round(preferred_skill * 10, 1),
             "detail": f"{len(preferred_shared)} of {len(preferred)} preferences evidenced" if preferred else "No preferred skills were specified"},
            {"label": "Relevant experience", "weight": 25, "evidence": round(exp * 100, 1),
             "points": round(exp * 25, 1),
             "detail": f"{candidate_years} years evidenced" + (f"; role requests {required_years}" if required_years else "; no minimum was specified")},
            {"label": "Qualifications", "weight": 10, "evidence": round(qual * 100, 1),
             "points": round(qual * 10, 1),
             "detail": "Matches the stated qualification" if required_education and qual == 1 else ("No qualification requirement was specified" if not required_education else "No matching qualification was evidenced")},
        ]
        improvement_plan = []
        if not required:
            improvement_plan.append("Ask the hiring team to list the role's must-have skills. Without them, this percentage is only a partial signal.")
        for missing_skill in missing:
            improvement_plan.append(f"For {missing_skill}: if you have used it, add a specific project, responsibility, and outcome to the resume. If not, gain real practice before claiming it.")
        for missing_skill in preferred_missing:
            improvement_plan.append(f"{missing_skill} is a preference, not a blocker. A relevant project or course can strengthen this part of the match.")
        if required_years and candidate_years < required_years:
            improvement_plan.append("Make related experience easy to verify: include dates, scope, tools used, and measurable outcomes. Do not inflate years of experience.")
        if required_education and qual < 1:
            improvement_plan.append("If you have an equivalent credential or relevant training, list it clearly. Otherwise, emphasize verified work samples and transferable experience.")
        if not improvement_plan:
            improvement_plan.append("Strengthen an already good match by adding measurable outcomes, role-relevant projects, and links to work samples.")
        evidence_found = []
        if shared:
            evidence_found.append(f"The resume explicitly mentions: {', '.join(shared)}.")
        if candidate_years:
            evidence_found.append(f"The resume states {candidate_years} years of experience; add dates and role context so a recruiter can verify the claim quickly.")
        if candidate.get("education"):
            evidence_found.append(f"The resume lists: {', '.join(candidate['education'])}.")
        if not evidence_found:
            evidence_found.append("No role-specific evidence could be extracted reliably. Use a text-based resume or add a clear Skills and Experience section.")
        resume_edits = []
        for skill_name in shared:
            resume_edits.append(f"For {skill_name}, add one bullet in this format: action + tool + measurable result (only if it is true).")
        if missing:
            resume_edits.append("Do not add missing skills as keywords alone. Either show a real project or leave them out and focus on adjacent experience.")
        if not candidate_years:
            resume_edits.append("Add month/year dates to each relevant role so experience can be evaluated accurately.")
        resume_edits.append("Lead each role with outcomes: what you built, the scale, and the result—not only a list of responsibilities.")
        growth_plan = []
        for skill_name in missing[:3]:
            growth_plan.append(f"Build a small, documented {skill_name} project; include a README, tests or screenshots, and a short explanation of your decisions.")
        if not growth_plan:
            growth_plan.append("Choose the highest-value requirement and deepen it with a portfolio case study that explains your trade-offs and results.")
        research_topics = sorted(set(missing + preferred_missing))[:5]
        score_interpretation = [
            f"The {score}% score is a weighted evidence score: it is not a prediction of job performance or an automated hiring decision.",
            f"Must-have skills contribute up to 55 points; this resume currently earns {round(skill * 55, 1)} points from that category.",
            f"Experience contributes up to 25 points; this resume currently earns {round(exp * 25, 1)} points based on explicitly stated years.",
        ]
        if missing:
            score_interpretation.append(f"The largest current score constraint is missing evidence for: {', '.join(missing)}.")
        elif required:
            score_interpretation.append("The core role requirements are evidenced; the strongest next improvement is to make the impact of that work easier to verify.")
        else:
            score_interpretation.append("The job brief does not state enough structured requirements for a high-confidence comparison.")
        resume_blueprint = [
            "Summary: state the target role, years of relevant experience, and two verified strengths that match the job.",
            "Skills: group only real skills under clear headings (for example, Languages, Frameworks, Cloud); do not use keyword stuffing.",
            "Experience: use 3–5 bullets per relevant role in this pattern: action + tool + scope + measurable outcome.",
            "Projects: add one role-relevant project with a concise README, your contribution, key technical choices, and a working link if available.",
        ]
        interview_preparation = [
            *(f"Prepare one STAR story that demonstrates {skill_name}: the situation, your actions, technical trade-offs, and measurable result." for skill_name in shared[:3]),
            "Be ready to explain the exact scope of your work, what you personally owned, and what you would improve next time.",
        ]
        if missing:
            interview_preparation.append("For a missing requirement, be direct about your current level and explain the concrete learning or project plan instead of overstating experience.")
        if missing:
            mentor_narrative = (f"Here is the honest read: your resume has some relevant evidence, but it does not yet show {', '.join(missing)}—"
                                f"important requirement{'s' if len(missing) > 1 else ''} for this role. "
                                "The fastest honest improvement is to make your existing relevant work specific, then build and document real evidence for the missing areas.")
        elif required:
            mentor_narrative = ("You are showing the core requirements on paper. The next step is not to add more buzzwords; "
                                "it is to make your contribution, scope, and results obvious within a few seconds of scanning the resume.")
        else:
            mentor_narrative = ("I can see parts of your background, but this job description is too vague to give a reliable fit assessment. "
                                "Ask for the must-have skills and success criteria, then tailor the resume to those verified requirements.")
        return {"score": score, "skill_relevance": round(skill*100,1),
          "experience_alignment": round(exp*100,1), "qualification_match": round(qual*100,1),
          "preferred_skill_match": round(preferred_skill*100,1), "recommendation": recommendation,
          "knockout_gaps": missing, "strengths": strengths or ["No stated must-haves evidenced"],
          "gaps": gaps, "score_breakdown": breakdown, "improvement_plan": improvement_plan,
          "deep_resume_review": {"evidence_found": evidence_found, "resume_edits": resume_edits,
                                 "growth_plan": growth_plan, "research_topics": research_topics},
          "detailed_explanation": {"score_interpretation": score_interpretation, "resume_blueprint": resume_blueprint,
                                   "interview_preparation": interview_preparation,
                                   "mentor_narrative": mentor_narrative,
                                   "evidence_limitations": "The analysis can only assess text successfully extracted from the uploaded document. Verify all recommendations against the original resume and job description."},
          "score_context": "The score reflects evidence in the uploaded resume, not a prediction of job performance.", "refinement_trace": trace,
          "disclaimer": "Decision support only. Review job-relevant evidence; do not use protected characteristics."}
