import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { ProcessingStatus } from "@/components/ProcessingStatus";
import { CaptionCard } from "@/components/CaptionCard";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { PROGRESS_MESSAGES } from "@/lib/mock-data";
import { useEffect, useState } from "react";
import type { Run } from "@/lib/types";
import { ArrowRight, Download, RefreshCw, Beaker } from "lucide-react";

type Search = { runId?: string };

export const Route = createFileRoute("/quick")({
  validateSearch: (s: Record<string, unknown>): Search => ({
    runId: typeof s.runId === "string" ? s.runId : undefined,
  }),
  component: QuickCaption,
});

type Phase = "idle" | "processing" | "results";

function QuickCaption() {
  const navigate = useNavigate();
  const { runId: initialRunId } = Route.useSearch();
  const [phase, setPhase] = useState<Phase>(initialRunId ? "processing" : "idle");
  const [step, setStep] = useState(0);
  const [run, setRun] = useState<Run | null>(null);
  const [runId, setRunId] = useState<string | undefined>(initialRunId);

  useEffect(() => {
    if (phase !== "processing") return;
    let cancelled = false;
    (async () => {
      const id = runId ?? (await api.createRun()).id;
      if (cancelled) return;
      setRunId(id);
      for (let i = 0; i < PROGRESS_MESSAGES.length; i++) {
        if (cancelled) return;
        setStep(i);
        await new Promise((r) => setTimeout(r, 780));
      }
      const r = await api.getRun(id);
      if (cancelled) return;
      setRun(r);
      setPhase("results");
    })();
    return () => {
      cancelled = true;
    };
  }, [phase, runId]);

  async function handleFile() {
    const r = await api.createRun();
    setRunId(r.id);
    setStep(0);
    setPhase("processing");
    navigate({ to: "/quick", search: { runId: r.id }, replace: true });
  }

  return (
    <div className="min-h-screen bg-background">
      <AppHeader />
      <main className="mx-auto max-w-4xl px-6 py-14">
        <div className="mb-10">
          <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            quick caption
          </div>
          <h1 className="mt-2 font-display text-5xl leading-tight tracking-tight text-balance">
            One video in.<br />
            <span className="text-ember">Grounded captions out.</span>
          </h1>
        </div>

        {phase === "idle" && (
          <div className="space-y-4">
            <UploadDropzone onFile={handleFile} />
            <p className="text-sm text-muted-foreground">
              Or use the{" "}
              <button
                onClick={handleFile}
                className="underline decoration-ember decoration-2 underline-offset-4"
              >
                sample clip
              </button>
              . No settings, no jargon — just captions.
            </p>
          </div>
        )}

        {phase === "processing" && (
          <ProcessingStatus
            steps={PROGRESS_MESSAGES}
            active={PROGRESS_MESSAGES[step]}
            pct={Math.round(((step + 1) / PROGRESS_MESSAGES.length) * 100)}
          />
        )}

        {phase === "results" && run && (
          <div className="space-y-8">
            <div className="grid gap-6 md:grid-cols-[1.4fr_1fr]">
              <div className="overflow-hidden rounded-xl border border-border bg-ink">
                <div className="flex aspect-video items-center justify-center bg-gradient-to-br from-[oklch(0.35_0.06_60)] to-ink text-paper/60">
                  <div className="text-center">
                    <div className="font-display text-3xl">{run.video.filename}</div>
                    <div className="mt-1 font-mono text-xs uppercase tracking-[0.2em] text-paper/50">
                      {run.video.width}×{run.video.height} · {run.video.durationSec}s · {run.video.codec}
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex flex-col justify-between rounded-xl border border-border bg-card p-5">
                <div>
                  <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
                    result
                  </div>
                  <div className="mt-1 font-display text-2xl">
                    {run.captions.results.length} captions ready
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Routed to{" "}
                    <span className="font-mono">{run.evidence.result.selectedRoute}</span>.
                    Grounded on visual scene, actions, and audio-safe facts.
                  </p>
                </div>
                <div className="mt-6 flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="gap-1.5"
                    onClick={() => {
                      const blob = new Blob([JSON.stringify(run, null, 2)], {
                        type: "application/json",
                      });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = `${run.id}.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                  >
                    <Download className="h-3.5 w-3.5" /> Download JSON
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="gap-1.5"
                    onClick={() => {
                      setPhase("processing");
                      setStep(0);
                    }}
                  >
                    <RefreshCw className="h-3.5 w-3.5" /> Generate again
                  </Button>
                </div>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              {run.captions.results.map((c) => (
                <CaptionCard key={c.id} caption={c} />
              ))}
            </div>

            <div className="rounded-xl border border-dashed border-ember/50 bg-ember-soft/40 p-5">
              <div className="flex flex-col items-start justify-between gap-3 md:flex-row md:items-center">
                <div>
                  <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
                    want to see how it worked?
                  </div>
                  <div className="mt-1 font-display text-2xl">Open this run in Gemma Lab</div>
                </div>
                <Button asChild className="gap-2">
                  <Link to="/lab/$runId/video" params={{ runId: run.id }}>
                    <Beaker className="h-4 w-4" /> Inspect pipeline <ArrowRight className="h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
