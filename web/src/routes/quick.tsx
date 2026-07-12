import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowRight, Beaker, Download, RefreshCw } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { ProcessingStatus } from "@/components/ProcessingStatus";
import { CaptionCard } from "@/components/CaptionCard";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/StateViews";
import { GenerationOutcomeNotice } from "@/components/GenerationOutcomeNotice";
import { api, mediaUrl, waitForRun } from "@/lib/api";
import { startQuickUpload } from "@/lib/quick-flow";
import type { Run, RunStatusResponse } from "@/lib/types";
import { ServiceHealthNotice } from "@/components/ServiceHealth";
import { useHealth } from "@/lib/hooks";

type Search = { runId?: string };
export const Route = createFileRoute("/quick")({ validateSearch: (s: Record<string, unknown>): Search => ({ runId: typeof s.runId === "string" ? s.runId : undefined }), component: QuickCaption });

const STEPS = ["Preparing video", "Selecting important moments", "Checking audio", "Understanding the scene with Gemma", "Writing captions with Gemma", "Complete"];
const STATUS_STEP: Record<string, string> = {
  video: "Preparing video",
  frames: "Selecting important moments",
  audio: "Checking audio",
  evidence: "Understanding the scene with Gemma",
  captions: "Writing captions with Gemma",
  compare: "Complete",
};

function QuickCaption() {
  const navigate = useNavigate();
  const { runId } = Route.useSearch();
  const [run, setRun] = useState<Run | null>(null);
  const [status, setStatus] = useState<RunStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const health = useHealth();
  const serviceUnavailable = health.isError || health.data?.status === "unavailable";
  const generationUnavailable = serviceUnavailable || health.isPending || health.data?.providersConfigured === false;

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setError(null);
    api.getRun(runId).then((current) => {
      if (current.status === "ready") return setRun(current);
      if (current.status === "error") throw new Error(current.error ?? "Processing failed.");
      return waitForRun(runId, { signal: controller.signal, onStatus: setStatus }).then(setRun);
    }).catch((cause: Error) => { if (cause.name !== "AbortError") setError(cause.message); });
    return () => controller.abort();
  }, [runId]);

  async function handleFile(file: File) {
    if (generationUnavailable) {
      setError("Gemma generation is not configured. Open Gemma Lab when the service is available, or retry the health check.");
      return;
    }
    setUploading(true); setError(null); setRun(null);
    try { await startQuickUpload(file, (id) => navigate({ to: "/quick", search: { runId: id }, replace: true })); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Upload failed."); }
    finally { setUploading(false); }
  }

  function download() {
    if (!run) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(run, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${run.id}.json`; anchor.click(); URL.revokeObjectURL(url);
  }

  const activeStep = status?.activeStage ? STATUS_STEP[status.activeStage] : status?.progressMessage && STEPS.includes(status.progressMessage) ? status.progressMessage : "Preparing video";

  return (
    <div className="min-h-[100dvh] bg-background">
      <AppHeader />
      <main className="mx-auto max-w-[1440px] px-6 py-12 lg:px-10 lg:py-16">
        <ServiceHealthNotice className="mb-8" />
        {!runId && !error && (
          <section className="mx-auto max-w-2xl text-center">
            <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">Quick Caption</div>
            <h1 className="mt-4 font-display text-5xl font-semibold tracking-[-0.055em] sm:text-7xl">One video in.<br /><span className="text-muted-foreground">Grounded captions out.</span></h1>
            <p className="mx-auto mt-5 max-w-lg text-muted-foreground">The Balanced pipeline handles frames, useful audio, evidence, and four caption styles for you.</p>
            <UploadDropzone className="mt-10 text-left" onFile={handleFile} disabled={Boolean(generationUnavailable)} />
          </section>
        )}

        {uploading && <p className="mx-auto mt-5 max-w-2xl font-mono text-xs uppercase tracking-[0.16em] text-muted-foreground" role="status">Uploading video securely</p>}

        {runId && !run && !error && (
          <section className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr] lg:items-start">
            <div className="glass-panel rounded-xl p-6">
              <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Quick run</div>
              <div className="mt-3 truncate font-mono text-sm text-foreground">{runId}</div>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">GemmaClip is preparing the source. Your next result will reuse these stored artifacts.</p>
            </div>
            <ProcessingStatus steps={STEPS} active={activeStep} />
          </section>
        )}

        {error && (
          <ErrorState title="Could not create captions" description={error} action={<Button variant="outline" onClick={() => navigate({ to: "/quick", search: {}, replace: true })}>Choose another video</Button>} />
        )}

        {run && (
          <div className="grid gap-8 lg:grid-cols-[minmax(280px,0.72fr)_minmax(0,1.28fr)] lg:items-start">
            <aside className="space-y-4 lg:sticky lg:top-24">
              <div className="overflow-hidden rounded-xl border border-white/10 bg-card shadow-[0_20px_60px_rgb(0_0_0_/_0.22)]">
                <video className="aspect-video w-full bg-ink object-contain" controls src={mediaUrl(run.id)} aria-label={`Uploaded video ${run.video.filename}`} />
                <div className="space-y-4 p-5">
                  <div><div className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Run summary</div><div className="mt-2 truncate font-mono text-xs">{run.video.filename}</div></div>
                  <div className="grid grid-cols-2 gap-3 border-t border-white/10 pt-4 text-xs"><Summary label="captions" value={String(run.captions.results.length)} /><Summary label="route" value={run.evidence.result.selectedRoute} /><Summary label="provider" value={run.evidence.result.routeProvider} /><Summary label="status" value={run.status} /></div>
                  <div className="flex flex-wrap gap-2 border-t border-white/10 pt-4"><Button variant="outline" size="sm" className="min-h-11 gap-2" onClick={download}><Download className="h-3.5 w-3.5" /> Download JSON</Button><Button variant="ghost" size="sm" className="min-h-11 gap-2" onClick={() => navigate({ to: "/quick", search: {}, replace: true })}><RefreshCw className="h-3.5 w-3.5" /> Start another</Button></div>
                </div>
              </div>
            </aside>
            <section className="space-y-6">
              <GenerationOutcomeNotice outcome={run.generationOutcome} />
              <div><div className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Generated captions</div><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.045em]">Grounded in what Gemma saw.</h1></div>
              <div className="grid gap-4">{run.captions.results.map((caption) => <CaptionCard key={caption.id} caption={caption} />)}</div>
              <div className="flex flex-wrap items-center justify-between gap-4 border-t border-white/10 pt-6"><p className="max-w-md text-sm text-muted-foreground">Open this completed run in Gemma Lab to inspect frames, audio routing, evidence, and caption settings.</p><Button asChild variant="outline" className="min-h-11 gap-2"><Link to="/lab/$runId/video" params={{ runId: run.id }}><Beaker className="h-4 w-4" /> Inspect in Gemma Lab <ArrowRight className="h-4 w-4" /></Link></Button></div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return <div><div className="font-mono text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div><div className="mt-1 truncate font-mono text-xs text-foreground">{value}</div></div>;
}
