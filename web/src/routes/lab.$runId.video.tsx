import { createFileRoute } from "@tanstack/react-router";
import { VideoMetadataPanel } from "@/components/VideoMetadata";
import { ConfigSection, Field } from "@/components/ConfigSection";
import { PrevNext } from "@/components/PrevNext";
import { useRun } from "@/lib/hooks";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { useEffect, useMemo, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Play, RotateCw } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { ProcessingPreset } from "@/lib/types";
import { ProcessingState, StageErrorState, StaleStageNotice } from "@/components/StateViews";

export const Route = createFileRoute("/lab/$runId/video")({
  component: VideoStage,
});

const PRESETS: { id: ProcessingPreset; label: string; note: string }[] = [
  { id: "fast", label: "Fast", note: "Six uniform frames, audio off, quickest turnaround." },
  { id: "balanced", label: "Balanced", note: "Hybrid frame selection, automatic audio, auto route." },
  { id: "maximum", label: "Maximum Detail", note: "Dense frames, prefer audio-visual when usable." },
  { id: "custom", label: "Custom", note: "Configure every stage yourself in the lab." },
];

function VideoStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [preset, setPreset] = useState<ProcessingPreset>(run?.preset ?? "balanced");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => { if (run) setPreset(run.preset); }, [run]);

  const posterGrad = useMemo(
    () => "bg-gradient-to-br from-[oklch(0.28_0.08_60)] via-[oklch(0.35_0.09_35)] to-ink",
    [],
  );

  async function runPreset() {
    setBusy(true);
    setError(null); setNotice(null);
    try {
      const updated = await api.postMetadata(runId, preset);
      qc.setQueryData(runKey(runId), updated);
      setNotice("Preset updated. Frames, Audio, Evidence, and Captions need to be run again.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Preset update failed safely.");
    } finally { setBusy(false); }
  }

  if (!run) return <ProcessingState />;
  if (run.stages.video === "active") return <ProcessingState description={run.progressMessage ?? "Inspecting video metadata."} />;
  if (run.stages.video === "error") return <StageErrorState stage="Video" description={run.stageErrors?.video ?? run.error ?? undefined} onRetry={runPreset} />;
  const dirty = preset !== run.preset;

  return (
    <div>
      <StageHeader
        eyebrow="stage 01 · video"
        title="Inspect the source"
        description="Confirm the file you're working with and pick a processing preset. Downstream stages will be seeded from this choice."
      />
      {run.stages.video === "invalidated" && <StaleStageNotice message="Video metadata is stale. Reapply a preset or re-probe the source." />}
      {notice && <div className="mb-4 rounded-lg border border-success/30 bg-success/5 px-3 py-2 text-sm text-success" role="status">{notice}</div>}
      {error && <div className="mb-4 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger" role="alert">{error}</div>}

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div className="space-y-6">
          <div className={`relative overflow-hidden rounded-xl border border-border ${posterGrad} aspect-video`}>
            <video className="absolute inset-0 h-full w-full bg-ink object-contain" controls src={mediaUrl(runId)} aria-label={`Uploaded video ${run.video.filename}`} />
            <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-ink/90 to-transparent px-5 py-4 font-mono text-xs text-paper/80">
              <span>{run.video.filename}</span>
              <span>{run.video.durationSec.toFixed(1)}s</span>
            </div>
          </div>

          <ConfigSection
            title="Processing preset"
          description="Inspect the preset that seeded this run."
            actions={
            <div className="flex flex-wrap gap-2">
              <Button variant="ghost" onClick={() => setPreset(run.preset)} disabled={busy || !dirty} size="sm">Reset</Button>
              <Button onClick={runPreset} disabled={busy || !dirty} size="sm" className="gap-1.5" title="Apply this preset and invalidate downstream stages">
              <RotateCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} /> Apply preset
              </Button>
            </div>
            }
          >
            <p className="mb-4 rounded-md border border-white/10 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
              {dirty ? "Unsaved configuration. Applying this preset invalidates downstream stages." : "Stored configuration. Choose a preset to prepare a new run."}
            </p>
            <RadioGroup
              value={preset}
              onValueChange={(v) => setPreset(v as ProcessingPreset)}
              className="grid gap-2 md:grid-cols-2"
            >
              {PRESETS.map((p) => (
                <label
                  key={p.id}
                  htmlFor={`preset-${p.id}`}
                  className={`flex items-start gap-3 rounded-lg border p-3 transition ${
                    preset === p.id
                      ? "border-ember bg-ember-soft"
                    : "border-white/10 bg-background hover:border-white/20"
                  }`}
                >
                  <RadioGroupItem id={`preset-${p.id}`} value={p.id} className="mt-0.5" />
                  <div>
                    <Label htmlFor={`preset-${p.id}`} className="cursor-pointer font-medium">
                      {p.label}
                    </Label>
                    <p className="mt-0.5 text-xs text-muted-foreground">{p.note}</p>
                  </div>
                </label>
              ))}
            </RadioGroup>
          </ConfigSection>
        </div>

        <div className="space-y-6">
          <VideoMetadataPanel meta={run.video} />
          <div className="rounded-xl border border-border bg-card p-5">
            <Field label="Run ID">
              <code className="rounded bg-muted px-2 py-1 font-mono text-xs">{run.id}</code>
            </Field>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <MiniStat k="preset" v={run.preset} />
              <MiniStat k="status" v={run.status} />
            </div>
          </div>
        </div>
      </div>

      <PrevNext
        runId={runId}
        next={{ to: "/lab/$runId/frames", label: "Frames" }}
      />
    </div>
  );
}

export function StageHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <header className="mb-7 max-w-3xl sm:mb-9">
      <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
        <span className="h-px w-8 bg-ember" aria-hidden="true" />
        {eyebrow}
      </div>
      <h1 className="mt-3 font-display text-3xl font-semibold leading-[1.03] tracking-[-0.035em] text-balance sm:text-4xl md:text-5xl">
        {title}
      </h1>
      {description && (
        <p className="mt-4 max-w-2xl text-pretty text-sm leading-7 text-muted-foreground sm:text-base">{description}</p>
      )}
    </header>
  );
}

function MiniStat({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg border border-border bg-background px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        {k}
      </div>
      <div className="mt-0.5 font-mono text-sm capitalize">{v}</div>
    </div>
  );
}
