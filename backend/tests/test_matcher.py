from app.services.matcher import RecursiveMatcher
from app.services.parser import parse_document
def test_recursive_match_explains_coverage_and_trace():
    result = RecursiveMatcher(4).match({"skills":["python","sql"],"years_experience":4,"education":["bachelor"]}, {"skills":["python","react"],"years_experience":3,"education":["bachelor"]})
    assert result["skill_relevance"] == 50
    assert "Missing requested skill: react" in result["gaps"]
    assert len(result["refinement_trace"]) == 4
    assert result["refinement_trace"][-1]["score"] > result["refinement_trace"][0]["score"]

def test_matcher_flags_missing_must_haves_for_recruiter_review():
    result = RecursiveMatcher().match(
        {"skills": ["python"], "years_experience": 1, "education": []},
        {"required_skills": ["python", "react", "sql"], "preferred_skills": ["aws"],
         "years_experience": 4, "education": ["bachelor"]},
    )
    assert result["score"] <= 49
    assert result["recommendation"] == "Does not currently meet most stated must-haves"
    assert result["knockout_gaps"] == ["react", "sql"]

def test_matcher_can_score_explicit_role_terms_not_in_the_builtin_skill_list():
    job = parse_document("Must have Laravel and Redis. Docker is preferred.")
    candidate = parse_document("Built Laravel APIs with Redis caching for a customer portal.")
    result = RecursiveMatcher().match(candidate, job)
    assert job["role_terms"] == ["laravel", "redis"]
    assert result["skill_relevance"] == 100
