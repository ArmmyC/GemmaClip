import { createFileRoute } from "@tanstack/react-router";
import { ConfigSection, Field } from "@/components/ConfigSection";
import { PrevNext } from "@/components/PrevNext";
import { FrameTimeline } from "@/components/FrameTimeline";
import { FrameCard } from "@/components/FrameCard";
import { useRun } from "@/lib/hooks";
import { StageHeader } from "./lab.$runId.video";
import { useEffect, useState } from "react";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Play, RotateCw } from "lucide-react";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { FrameMethod, FrameRequest } from "@/lib/types";
import { InvalidationPreview, ProcessingState, StageErrorState, StaleStageNotice } from "@/components/StateViews";

export const Route = createFileRoute("/lab/$runId/frames")({
  component: FramesStage,
});

const METHODS: { id: FrameMethod; label: string; note: string }[] = [
  { id: "uniform", label: "Uniform", note: "Fixed spacing across the clip." },
  { id: "aks-lite", label: "AKS-Lite", note: "Adaptive Keyframe Sampling, driven by visual change." },
  { id: "hybrid", label: "Hybrid: Anchors + AKS-Lite", note: "Guaranteed anchors + top change frames." },
];

function FramesStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [method, setMethod] = useState<FrameMethod>(run?.frames.config.method ?? "hybrid");
  const [total, setTotal] = useState(run?.frames.config.totalFrames ?? 6);
  const [anchors, setAnchors] = useState(run?.frames.config.anchorCount ?? 2);
  const [high, setHigh] = useState(run?.frames.config.highChangeCount ?? 3);
  const [spacing, setSpacing] = useState(run?.frames.config.minSpacingSec ?? 1.5);
  const [sens, setSens] = useState(run?.frames.config.changeSensitivity ?? 0.55);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    if (!run) return;
    setMethod(run.frames.config.method); setTotal(run.frames.config.totalFrames); setAnchors(run.frames.config.anchorCount);
    setHigh(run.frames.config.highChangeCount); setSpacing(run.frames.config.minSpacingSec); setSens(run.frames.config.changeSensitivity);
  }, [run]);

  if (!run) return <ProcessingState />;
  if (run.stages.frames === "active") return <ProcessingState description={run.progressMessage ?? "Selecting important moments."} />;
  if (run.stages.frames === "error") return <StageErrorState stage="Frames" description={run.stageErrors?.frames ?? run.error ?? undefined} onRetry={() => extract()} />;
  const draft: FrameRequest = { method, totalFrames: total, anchorCount: anchors, highChangeCount: high, minSpacingSec: spacing, changeSensitivity: sens };
  const dirty = JSON.stringify(draft) !== JSON.stringify(run.frames.config);

  async function extract() {
    setBusy(true);
    setError(null); setNotice(null);
    try {
      const updated = await api.postFrames(runId, draft);
      qc.setQueryData(runKey(runId), updated);
      setNotice("Frames updated. Evidence, Captions, and Compare require regeneration.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Frame extraction failed safely.");
    } finally { setBusy(false); }
  }

  async function toggleFrame(id: string, included: boolean) {
    if (!run) return;
    const selected = run.frames.frames.filter((frame) => frame.id === id ? included : frame.included).map((frame) => frame.id);
    try {
      const updated = await api.postFrameSelection(runId, { includedFrameIds: selected });
      qc.setQueryData(runKey(runId), updated);
      setNotice("Frame selection saved. Evidence, Captions, and Compare are stale until regenerated.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Frame selection could not be saved.");
      throw cause;
    }
  }

  return (
    <div>
      <StageHeader
        eyebrow="stage 02 · frames"
        title="Select what Gemma sees"
        description="Frame selection decides the visual evidence budget. Anchors guarantee coverage; AKS-Lite catches motion; Hybrid combines both."
      />
      {run.stages.frames === "invalidated" && <StaleStageNotice />}
      {dirty && <InvalidationPreview stage="frames" />}
      {notice && <div className="mb-4 rounded-lg border border-success/30 bg-success/5 px-3 py-2 text-sm text-success" role="status">{notice}</div>}
      {error && <div className="mb-4 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger" role="alert">{error}</div>}

      <div className="grid gap-6 lg:grid-cols-[1fr_1.6fr]">
        <ConfigSection
          title="Extraction"
          description="Configure how many moments Gemma should see."
          actions={
            <div className="flex flex-wrap gap-2">
              <Button variant="ghost" size="sm" onClick={() => { setMethod(run.frames.config.method); setTotal(run.frames.config.totalFrames); setAnchors(run.frames.config.anchorCount); setHigh(run.frames.config.highChangeCount); setSpacing(run.frames.config.minSpacingSec); setSens(run.frames.config.changeSensitivity); }} disabled={busy || !dirty}>Reset</Button>
              <Button size="sm" className="gap-1.5" onClick={extract} disabled={busy || run.stages.video !== "complete" || (!dirty && run.stages.frames === "complete")}>
              {busy ? (
                <>
                  <RotateCw className="h-3.5 w-3.5 animate-spin" /> extracting
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5" /> Extract frames
                </>
              )}
              </Button>
            </div>
          }
        >
          <p className="rounded-md border border-white/10 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
            {dirty ? "Unsaved configuration. Evidence, Captions, and Compare will be invalidated." : "Stored configuration. Adjust a setting, then run frames."}
          </p>
          <RadioGroup value={method} onValueChange={(v) => setMethod(v as FrameMethod)} className="grid gap-2">
            {METHODS.map((m) => (
              <label
                key={m.id}
                htmlFor={`m-${m.id}`}
                className={`flex items-start gap-3 rounded-lg border p-3 transition ${
                  method === m.id
                    ? "border-ember bg-ember-soft"
                    : "border-white/10 bg-background hover:border-white/20"
                }`}
              >
                <RadioGroupItem id={`m-${m.id}`} value={m.id} className="mt-0.5" />
                <div>
                  <Label htmlFor={`m-${m.id}`} className="cursor-pointer font-medium">
                    {m.label}
                  </Label>
                  <p className="mt-0.5 text-xs text-muted-foreground">{m.note}</p>
                </div>
              </label>
            ))}
          </RadioGroup>

          <Field label="Total frames" hint={<span className="font-mono">{total}</span>}>
              <Slider value={[total]} onValueChange={([v]) => { setTotal(v); setAnchors((current) => Math.min(current, v)); setHigh((current) => Math.min(current, Math.max(0, v - Math.min(anchors, v)))); }} min={6} max={16} step={1} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Anchors" hint={<span className="font-mono">{anchors}</span>}>
              <Slider value={[anchors]} onValueChange={([v]) => setAnchors(Math.min(v, total - high))} min={0} max={6} step={1} />
            </Field>
            <Field label="High change" hint={<span className="font-mono">{high}</span>}>
              <Slider value={[high]} onValueChange={([v]) => setHigh(Math.min(v, total - anchors))} min={0} max={8} step={1} />
            </Field>
          </div>
          <Field label="Min spacing" hint={<span className="font-mono">{spacing.toFixed(1)}s</span>}>
            <Slider value={[spacing]} onValueChange={([v]) => setSpacing(v)} min={0.2} max={5} step={0.1} />
          </Field>
          <Field label="Change sensitivity" hint={<span className="font-mono">{sens.toFixed(2)}</span>}>
            <Slider value={[sens]} onValueChange={([v]) => setSens(v)} min={0} max={1} step={0.01} />
          </Field>
        </ConfigSection>

        <div className="space-y-5">
          <FrameTimeline
            series={run.frames.changeSeries}
            frames={run.frames.frames}
            durationSec={run.video.durationSec}
          />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {run.frames.frames.map((f) => (
              <FrameCard key={f.id} frame={f} onToggle={toggleFrame} />
            ))}
          </div>
        </div>
      </div>

      <PrevNext
        runId={runId}
        prev={{ to: "/lab/$runId/video", label: "Video" }}
        next={{ to: "/lab/$runId/audio", label: "Audio" }}
      />
    </div>
  );
}
