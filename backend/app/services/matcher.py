"""JAX CPU reference implementation of SmartHire's recursive matcher.

This deliberately small, interpretable model uses a single recurrence across all
passes. A trained ~7M-parameter JAX/Flax TRM can replace `recursive_scores`
without changing this class's output contract.
"""
from typing import Any
import jax
import jax.numpy as jnp

@jax.jit
def recursive_scores(evidence: jax.Array, passes: int = 4) -> jax.Array:
    """Apply the same state update repeatedly and return every state.

    `evidence` is [skill coverage, experience alignment, qualification match].
    The weights make each factor explicit; in a trained TRM they would be learned.
    """
    weighted_evidence = jnp.dot(evidence, jnp.array([0.60, 0.25, 0.15]))
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
        required, offered = set(job["skills"]), set(candidate["skills"])
        shared, missing = sorted(required & offered), sorted(required - offered)
        skill = len(shared) / max(1, len(required))
        exp = min(1.0, candidate["years_experience"] / max(1, job["years_experience"])) if job["years_experience"] else .5
        qual = 1.0 if set(candidate["education"]) & set(job["education"]) else .5
        evidence = jnp.array([skill, exp, qual], dtype=jnp.float32)
        states = recursive_scores(evidence, self.passes).tolist()
        weighted = float(jnp.dot(evidence, jnp.array([.60, .25, .15])))
        trace = [{"pass": step, "score": round(float(state) * 100, 1),
                  "evidence": round(weighted * 100, 1)} for step, state in enumerate(states, 1)]
        return {"score": trace[-1]["score"], "skill_relevance": round(skill*100,1),
          "experience_alignment": round(exp*100,1), "qualification_match": round(qual*100,1),
          "strengths": [f"Matches required skill: {s}" for s in shared] or ["No listed required skills matched"],
          "gaps": [f"Missing requested skill: {s}" for s in missing], "refinement_trace": trace,
          "disclaimer": "Decision support only; recruiter review is required."}
