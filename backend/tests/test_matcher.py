from app.services.matcher import RecursiveMatcher
def test_recursive_match_explains_coverage_and_trace():
    result = RecursiveMatcher(4).match({"skills":["python","sql"],"years_experience":4,"education":["bachelor"]}, {"skills":["python","react"],"years_experience":3,"education":["bachelor"]})
    assert result["skill_relevance"] == 50
    assert "Missing requested skill: react" in result["gaps"]
    assert len(result["refinement_trace"]) == 4
    assert result["refinement_trace"][-1]["score"] > result["refinement_trace"][0]["score"]
