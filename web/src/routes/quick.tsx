import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowRight, Beaker, Download, RefreshCw } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { ProcessingStatus } from "@/components/ProcessingStatus";
import { CaptionCard } from "@/components/CaptionCard";
import { Button } from "@/components/ui/button";
import { api, mediaUrl, waitForRun } from "@/lib/api";
import type { Run, RunStatusResponse } from "@/lib/types";

type Search = { runId?: string };
export const Route = createFileRoute("/quick")({ validateSearch: (s: Record<string, unknown>): Search => ({ runId: typeof s.runId === "string" ? s.runId : undefined }), component: QuickCaption });
const STEPS = ["Inspecting video", "Selecting important moments", "Checking audio", "Building grounded evidence", "Writing captions", "Complete"];

function QuickCaption() {
  const navigate = useNavigate(); const { runId } = Route.useSearch();
  const [run, setRun] = useState<Run | null>(null); const [status, setStatus] = useState<RunStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null); const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (!runId) return; const controller = new AbortController(); setError(null);
    api.getRun(runId).then((current) => {
      if (current.status === "ready") return setRun(current);
      if (current.status === "error") throw new Error(current.error ?? "Processing failed.");
      return waitForRun(runId, { signal: controller.signal, onStatus: setStatus }).then(setRun);
    }).catch((e: Error) => { if (e.name !== "AbortError") setError(e.message); });
    return () => controller.abort();
  }, [runId]);

  async function handleFile(file: File) {
    setUploading(true); setError(null); setRun(null);
    try { const created = await api.createRun(file); await api.startQuickCaption(created.id); await navigate({ to: "/quick", search: { runId: created.id }, replace: true }); }
    catch (e) { setError(e instanceof Error ? e.message : "Upload failed."); } finally { setUploading(false); }
  }

  function download() { if (!run) return; const url = URL.createObjectURL(new Blob([JSON.stringify(run, null, 2)], { type: "application/json" })); const a = document.createElement("a"); a.href = url; a.download = `${run.id}.json`; a.click(); URL.revokeObjectURL(url); }

  return <div className="min-h-screen bg-background"><AppHeader /><main className="mx-auto max-w-4xl px-6 py-14">
    <div className="mb-10"><div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">quick caption</div><h1 className="mt-2 font-display text-5xl leading-tight">One video in.<br/><span className="text-ember">Grounded captions out.</span></h1></div>
    {!runId && <UploadDropzone onFile={handleFile} />}
    {uploading && <p className="mt-4 text-sm text-muted-foreground">Uploading video securely…</p>}
    {runId && !run && !error && <ProcessingStatus steps={STEPS} active={status?.progressMessage ?? "Inspecting video"} />}
    {error && <div role="alert" className="rounded-xl border border-destructive/40 bg-card p-5"><h2 className="font-display text-2xl">Could not create captions</h2><p className="mt-2 text-sm text-muted-foreground">{error}</p><Button className="mt-4" variant="outline" onClick={() => navigate({ to: "/quick", search: {}, replace: true })}>Choose another video</Button></div>}
    {run && <div className="space-y-8"><div className="grid gap-6 md:grid-cols-[1.4fr_1fr]"><video className="aspect-video w-full rounded-xl border border-border bg-ink" controls src={mediaUrl(run.id)} aria-label={`Uploaded video ${run.video.filename}`} /><div className="flex flex-col justify-between rounded-xl border border-border bg-card p-5"><div><div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">result</div><div className="mt-1 font-display text-2xl">{run.captions.results.length} captions ready</div><p className="mt-2 text-sm text-muted-foreground">Grounded through {run.evidence.result.selectedRoute}.</p></div><div className="mt-6 flex gap-2"><Button variant="outline" size="sm" onClick={download}><Download className="mr-1 h-3.5 w-3.5"/>Download JSON</Button><Button variant="ghost" size="sm" onClick={() => navigate({ to: "/quick", search: {}, replace: true })}><RefreshCw className="mr-1 h-3.5 w-3.5"/>Generate again</Button></div></div></div>
      <div className="grid gap-4 md:grid-cols-2">{run.captions.results.map(c => <CaptionCard key={c.id} caption={c}/>)}</div>
      <div className="rounded-xl border border-dashed border-ember/50 bg-ember-soft/40 p-5"><div className="flex items-center justify-between gap-3"><div><div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">want to see how it worked?</div><div className="mt-1 font-display text-2xl">Open this run in Gemma Lab</div></div><Button asChild><Link to="/lab/$runId/video" params={{ runId: run.id }}><Beaker className="mr-2 h-4 w-4"/>Inspect pipeline <ArrowRight className="ml-2 h-4 w-4"/></Link></Button></div></div>
    </div>}
  </main></div>;
}
