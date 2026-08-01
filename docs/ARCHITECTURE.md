# SmartHire AI architecture

```text
Recruiter dashboard
  | uploads PDF/DOCX resumes + job text/file
  v
FastAPI /jobs endpoint -> document extractor -> deterministic field parser
  |                                  (skills, years, education, work text)
  v
hashed structured embeddings (skill / experience / qualification channels)
  v
RecursiveMatcher: initial state -> 4 shared-weight refinement passes
  | each pass blends evidence, coverage, and prior state
  v
persist Job, Candidate, MatchScore -> ranked, evidence-backed JSON -> dashboard
```

The server is intentionally CPU-first: hashed vectors are 256 dimensions and matching requires no remote model. The supplied matcher is a transparent **reference/stub**, not a claim of a trained 7M-parameter TRM. It preserves the essential recursive property: each pass reuses the same update rule and incorporates the previous assessment, so the score trajectory is available to the reviewer.

## Production TRM path

Train a ~7M **JAX/Flax** shared-weight recursive encoder with labelled `(resume, JD, recruiter outcome)` pairs. Strip names, contact details, age/date proxies and protected attributes before training. Split by job and time, calibrate probabilities, compare to non-recursive baselines, audit subgroup outcomes where lawful, and version model/data/evaluation reports. Replace the `recursive_scores` function / `RecursiveMatcher` behind `MatcherProtocol`; retain the output contract (`score`, factors, strengths, gaps, refinement trace). Human approval remains mandatory.

## Supabase schema

`backend/app/models.py` maps `jobs`, `candidates`, and `match_scores`. Set `DATABASE_URL` to Supabase's Postgres URL and run `Base.metadata.create_all` for a prototype; use Alembic migrations before production. Files are parsed in-process in this scaffold; production should store originals in private object storage and retain only scoped links.

## Guardrails

Do not score or infer race, religion, sex, disability, age, nationality, or other protected data. Do not use the rank as a rejection rule. Limit retention, encrypt uploads, implement access controls/audit logs, and offer a review/appeal process.
