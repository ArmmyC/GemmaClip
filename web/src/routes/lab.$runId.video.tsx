import { createFileRoute } from "@tanstack/react-router";
import { VideoMetadataPanel } from "@/components/VideoMetadata";
import { ConfigSection, Field } from "@/components/ConfigSection";
import { PrevNext } from "@/components/PrevNext";
import { useRun } from "@/lib/hooks";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { useMemo, useState } from "react";
import { api, mediaUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Play, RotateCw } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { ProcessingPreset } from "@/lib/types";
import { ProcessingState } from "@/components/StateViews";

export const Route = createFileRoute("/lab/$runId/video")({
  component: VideoStage,
});

const PRESETS: { id: ProcessingPreset; label: string; note: string }[] = [
  { id: "fast", label: "Fast", note: "Fewer frames, visual-only route, quickest turnaround." },
  { id: "balanced", label: "Balanced", note: "Hybrid frame selection, automatic audio, auto route." },
  { id: "maximum", label: "Maximum Detail", note: "Dense frames, mandatory audio, unified route." },
  { id: "custom", label: "Custom", note: "Configure every stage yourself in the lab." },
];

function VideoStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [preset, setPreset] = useState<ProcessingPreset>(run?.preset ?? "balanced");
  const [busy, setBusy] = useState(false);

  const posterGrad = useMemo(
    () => "bg-gradient-to-br from-[oklch(0.28_0.08_60)] via-[oklch(0.35_0.09_35)] to-ink",
    [],
  );

  async function runPreset() {
    setBusy(true);
    const updated = await api.postMetadata(runId, preset);
    qc.setQueryData(runKey(runId), updated);
    setBusy(false);
  }

  if (!run || run.stages.video !== "complete") return <ProcessingState />;

  return (
    <div>
      <StageHeader
        eyebrow="stage 01 · video"
        title="Inspect the source"
        description="Confirm the file you're working with and pick a processing preset. Downstream stages will be seeded from this choice."
      />

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
              <Button onClick={runPreset} disabled size="sm" className="gap-1.5" title="Interactive reruns coming in the next integration phase">
                <RotateCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} /> Apply preset
              </Button>
            }
          >
            <p className="mb-4 rounded-md border border-white/10 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
              Configuration preview. Applying a new preset is not available for this run yet.
            </p>
            <RadioGroup
              disabled
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
                      : "border-white/10 bg-background"
                  }`}
                >
                  <RadioGroupItem disabled id={`preset-${p.id}`} value={p.id} className="mt-0.5" />
                  <div>
                    <Label htmlFor={`preset-${p.id}`} className="font-medium">
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
    <header className="mb-8 max-w-3xl">
      <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        <span className="h-px w-6 bg-ember" aria-hidden="true" />
        {eyebrow}
      </div>
      <h1 className="mt-3 font-display text-4xl font-semibold leading-[1.02] tracking-[-0.045em] text-balance md:text-5xl">
        {title}
      </h1>
      {description && (
        <p className="mt-3 text-pretty text-muted-foreground">{description}</p>
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
