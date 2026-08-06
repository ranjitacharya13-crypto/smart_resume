'use client';
import { type ChangeEvent, useMemo, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
type Breakdown = { label: string; weight: number; points: number; detail: string };
type Review = { evidence_found: string[]; resume_edits: string[]; growth_plan: string[]; research_topics: string[] };
type Details = { score_interpretation: string[]; resume_blueprint: string[]; interview_preparation: string[]; mentor_narrative: string; evidence_limitations: string };
type Candidate = { id: number; name: string; score: number; strengths: string[]; gaps: string[]; skill_relevance: number; experience_alignment: number; qualification_match: number; recommendation: string; score_context: string; score_breakdown: Breakdown[]; deep_resume_review: Review; detailed_explanation: Details; refinement_trace: { pass: number; score: number }[] };

const emptyReview = { evidence_found: [], resume_edits: [], growth_plan: [], research_topics: [] };

export default function Home() {
  const [job, setJob] = useState<number | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [list, setList] = useState<Candidate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [statusMessage, setStatusMessage] = useState('Drop in resumes and let the constellation reveal the strongest fit.');

  const fileSummary = useMemo(() => {
    if (!files || files.length === 0) return 'No resumes selected yet';
    return `${files.length} resume${files.length > 1 ? 's' : ''} detected`;
  }, [files]);

  async function analyzeSelected(selectedFiles: FileList | null) {
    if (!selectedFiles?.length) {
      setError('Upload at least one resume to begin.');
      return;
    }

    try {
      setBusy(true);
      setError('');
      setStatusMessage('Reading the documents and mapping the strongest signals...');

      const title = 'Resume screening';
      const description = `Auto-detected from uploaded resumes: ${Array.from(selectedFiles).map((file) => file.name).join(', ')}`;

      const jd = new FormData();
      jd.append('title', title);
      jd.append('description', description);

      let response = await fetch(`${API}/jobs`, { method: 'POST', body: jd });
      if (!response.ok) throw new Error(await response.text());

      const jobData = await response.json();
      setJob(jobData.id);

      const uploadForm = new FormData();
      Array.from(selectedFiles).forEach((file) => uploadForm.append('files', file));
      response = await fetch(`${API}/jobs/${jobData.id}/resumes`, { method: 'POST', body: uploadForm });
      if (!response.ok) throw new Error(await response.text());

      response = await fetch(`${API}/jobs/${jobData.id}/shortlist`);
      if (!response.ok) throw new Error(await response.text());

      const payload = await response.json();
      setList(payload.candidates || []);
      setStatusMessage(payload.candidates?.length ? 'The constellation is complete — here is the guided shortlist.' : 'The resumes were reviewed and mapped, but no candidates surfaced yet.');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed');
    } finally {
      setBusy(false);
    }
  }

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = event.target.files;
    setFiles(selectedFiles);
    if (selectedFiles?.length) {
      setStatusMessage(`Resume detected — ${selectedFiles.length} document${selectedFiles.length > 1 ? 's' : ''} prepared for review.`);
      void analyzeSelected(selectedFiles);
    }
  }

  return (
    <main className="art-shell min-h-screen px-4 py-8 text-slate-800 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <nav className="art-panel flex items-center justify-between px-5 py-3.5">
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-gradient-to-br from-violet-500 via-fuchsia-500 to-sky-500 text-sm font-black text-white shadow-lg shadow-violet-200">R</span>
            <div>
              <p className="text-lg font-black tracking-tight text-slate-900">ResumeScreening.ai</p>
              <p className="text-xs font-medium text-slate-500">Talent constellation map</p>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-violet-100 bg-violet-50/80 px-3 py-1.5 text-xs font-semibold text-violet-700 sm:flex">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Mentor-led review ready
          </div>
        </nav>

        <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="art-panel p-7 sm:p-8">
            <p className="mono text-[11px] font-semibold uppercase tracking-[0.3em] text-violet-600">Ethereal hiring lens</p>
            <h1 className="mt-3 text-4xl font-black leading-[1.05] tracking-tight text-slate-900 sm:text-5xl">
              Let each resume become a <span className="bg-gradient-to-r from-violet-600 via-fuchsia-500 to-sky-600 bg-clip-text text-transparent">clear signal</span>.
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
              Upload a candidate file and let the workspace read the experience, strengths, and growth edges with a calm, professional voice—like a mentor guiding you to the right next move.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <Badge text="Auto-detected resumes" />
              <Badge text="Mentor guidance" />
              <Badge text="Lightweight review" />
            </div>

            <div className="mt-8 rounded-[24px] border border-violet-100 bg-gradient-to-br from-white via-violet-50/70 to-sky-50 p-4 shadow-sm">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-700">Current signal</p>
                  <p className="text-sm text-slate-500">{statusMessage}</p>
                </div>
                <span className="rounded-full border border-violet-100 bg-white/90 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-violet-700">
                  {fileSummary}
                </span>
              </div>
            </div>
          </div>

          <div className="art-panel p-6 sm:p-7">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="mono text-[11px] font-semibold uppercase tracking-[0.3em] text-slate-500">Workspace</p>
                <h2 className="mt-1 text-2xl font-black tracking-tight text-slate-900">Upload and review</h2>
              </div>
              <div className="rounded-2xl border border-violet-100 bg-violet-50/80 px-3 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-violet-700">
                Instant read
              </div>
            </div>

            <label className="mt-6 flex min-h-[250px] cursor-pointer flex-col items-center justify-center rounded-[28px] border border-dashed border-violet-200 bg-gradient-to-br from-white via-violet-50/80 to-sky-50 px-6 py-8 text-center transition hover:border-violet-400 hover:shadow-lg hover:shadow-violet-100">
              <input className="sr-only" type="file" multiple accept=".pdf,.docx,.txt" onChange={handleFileSelection} />
              <div className="grid h-14 w-14 place-items-center rounded-2xl bg-white text-2xl shadow-sm">☁</div>
              <p className="mt-4 text-lg font-semibold text-slate-800">Drop resumes here</p>
              <p className="mt-2 max-w-sm text-sm leading-6 text-slate-500">The system will detect the resume content, read the narrative, and explain the fit like a seasoned mentor.</p>
            </label>

            <button disabled={busy || !files?.length} onClick={() => void analyzeSelected(files)} className="art-button mt-5 w-full">
              {busy ? <><Spinner /> Reading resumes…</> : <>Reveal the shortlist</>}
            </button>

            {error ? <p role="alert" className="mt-4 rounded-2xl border border-rose-200 bg-rose-50/80 px-3 py-2 text-sm font-medium text-rose-700">{error}</p> : null}
          </div>
        </section>

        {job ? (
          <section className="art-panel p-6 sm:p-7">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="mono text-[11px] font-semibold uppercase tracking-[0.3em] text-violet-600">Starlight results</p>
                <h2 className="mt-1 text-3xl font-black tracking-tight text-slate-900">Shortlisted candidates</h2>
              </div>
              <div className="rounded-full border border-violet-100 bg-violet-50/80 px-3 py-1.5 text-sm font-semibold text-violet-700">
                {list.length} reviewed • mentor notes ready
              </div>
            </div>

            {list.length ? (
              <div className="mt-6 grid gap-4 xl:grid-cols-2">
                {list.map((candidate, index) => (
                  <CandidateCard key={candidate.id} c={candidate} i={index} />
                ))}
              </div>
            ) : (
              <div className="mt-6 rounded-[24px] border border-dashed border-violet-200 bg-violet-50/40 p-8 text-center text-sm text-slate-500">
                The analysis has started. The shortlist will appear here as soon as the resume signals are mapped.
              </div>
            )}
          </section>
        ) : null}
      </div>
    </main>
  );
}

function Badge({ text }: { text: string }) {
  return <span className="art-pill">✦ {text}</span>;
}

function Spinner() {
  return <i className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />;
}

function CandidateCard({ c, i }: { c: Candidate; i: number }) {
  const details = c.detailed_explanation;
  const review = c.deep_resume_review || emptyReview;

  return (
    <article className="rounded-[28px] border border-violet-100 bg-gradient-to-br from-white via-violet-50/70 to-sky-50 p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-violet-500">{String(i + 1).padStart(2, '0')} · {c.recommendation}</p>
          <h3 className="mt-1 text-xl font-black text-slate-900">{c.name}</h3>
        </div>
        <div className="rounded-[20px] border border-violet-100 bg-white/90 px-4 py-3 text-center shadow-sm">
          <div className="text-3xl font-black text-slate-900">{c.score}%</div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.24em] text-violet-500">Signal strength</div>
        </div>
      </div>

      <div className="mt-5 rounded-[20px] border border-white/80 bg-white/80 p-4 text-sm leading-6 text-slate-600">
        {c.score_context}
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-3">
        <Metric n="Skills" v={c.skill_relevance} />
        <Metric n="Experience" v={c.experience_alignment} />
        <Metric n="Qualifications" v={c.qualification_match} />
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2">
        <Insight title="Strengths" items={c.strengths} good />
        <Insight title="Needs attention" items={c.gaps} />
      </div>

      <div className="mt-5 rounded-[20px] border border-violet-100 bg-white/80 p-4">
        <p className="text-sm font-semibold text-slate-700">Mentor guidance</p>
        <ul className="mt-3 space-y-2 text-sm text-slate-600">
          {(details?.resume_blueprint?.slice(0, 2) || []).map((item) => <li key={item}>• {item}</li>)}
          {(details?.interview_preparation?.slice(0, 2) || []).map((item) => <li key={item}>• {item}</li>)}
          {!details?.resume_blueprint?.length && !details?.interview_preparation?.length ? <li>• The profile appears strong, but a sharper story and tailored examples would make it easier to trust.</li> : null}
        </ul>
      </div>

      <details className="mt-4 rounded-[20px] border border-violet-100 bg-white/80 p-3 text-sm shadow-sm">
        <summary className="cursor-pointer font-semibold text-violet-700">Read the professional breakdown</summary>
        <div className="mt-3 space-y-3 text-slate-600">
          <p>{details?.mentor_narrative}</p>
          <Coach title="Evidence gathered" items={review.evidence_found} />
          <Coach title="Resume edits to make" items={review.resume_edits} />
          <Coach title="Growth plan" items={review.growth_plan} />
          <Coach title="Research topics" items={review.research_topics} />
        </div>
      </details>
    </article>
  );
}

function Metric({ n, v }: { n: string; v: number }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs font-semibold text-slate-500">
        <span>{n}</span>
        <span className="text-slate-700">{v}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200">
        <div className="h-full rounded-full bg-gradient-to-r from-violet-500 via-fuchsia-500 to-sky-500" style={{ width: `${v}%` }} />
      </div>
    </div>
  );
}

function Insight({ title, items, good = false }: { title: string; items: string[]; good?: boolean }) {
  return (
    <div>
      <h4 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">{title}</h4>
      <div className="flex flex-wrap gap-2">
        {items.length ? items.map((item) => (
          <span key={item} className={`rounded-full px-2.5 py-1 text-xs font-semibold ${good ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'}`}>
            {good ? '✓' : '!'} {item}
          </span>
        )) : <span className="text-xs text-slate-500">No notes surfaced here.</span>}
      </div>
    </div>
  );
}

function Coach({ title, items }: { title: string; items: string[] }) {
  return items.length ? (
    <div>
      <h4 className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">{title}</h4>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
        {items.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  ) : null;
}
