import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowLeftRight, Save, Beaker } from "lucide-react";
import { api, type ExperimentComparison } from "@/lib/api";
import { useRun } from "@/lib/hooks";
import { StageHeader } from "./lab.$runId.video";
import { PrevNext } from "@/components/PrevNext";
import { Button } from "@/components/ui/button";
import { ProcessingState, StageErrorState, StaleStageNotice } from "@/components/StateViews";
import type { Experiment } from "@/lib/types";
import { cn } from "@/lib/utils";
import { runKey } from "@/lib/hooks";

export const Route = createFileRoute("/lab/$runId/compare")({ component: CompareStage });

function CompareStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null);
  const [label, setLabel] = useState("");
  const [style, setStyle] = useState("formal");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!run?.experiments.length) return;
    setLeftId((value) => value || run.experiments[0].id);
    setRightId((value) => value || run.experiments[1]?.id || run.experiments[0].id);
  }, [run?.experiments]);

  useEffect(() => {
    if (!leftId || !rightId || leftId === rightId) { setComparison(null); return; }
    api.compareExperiments(runId, leftId, rightId).then(setComparison).catch(() => setComparison(null));
  }, [runId, leftId, rightId]);

  if (!run) return <ProcessingState />;
  if (run.stages.compare === "active") return <ProcessingState description="Saving or comparing experiment snapshots." />;
  if (run.stages.compare === "error") return <StageErrorState stage="Compare" description={run.stageErrors?.compare ?? run.error ?? undefined} />;

  async function saveExperiment() {
    setBusy(true); setError(null); setNotice(null);
    try {
      const updated = await api.postExperiment(runId, { label: label.trim() || undefined, captionStyle: style });
      qc.setQueryData(runKey(runId), updated);
      setLabel(""); setNotice("Experiment snapshot saved. Choose it below to compare.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment could not be saved.");
    } finally { setBusy(false); }
  }

  const hasCaptions = run.stages.captions === "complete";
  return (
    <div>
      <StageHeader eyebrow="stage 06 · compare" title="Compare experiments" description="Save immutable snapshots of the current run, then compare real configurations, providers, runtimes, and captions." />
      {!hasCaptions && <StaleStageNotice message="Generate current captions before saving an experiment." />}
      {run.stages.compare === "invalidated" && <StaleStageNotice message="Comparison is stale because the active run changed. Saved snapshots remain immutable." />}
      {notice && <div className="my-4 rounded-lg border border-success/30 bg-success/5 px-3 py-2 text-sm text-success" role="status">{notice}</div>}
      {error && <div className="my-4 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger" role="alert">{error}</div>}

      <section className="glass-panel mb-6 rounded-xl p-5" aria-labelledby="save-experiment-heading">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-[220px] flex-1"><label id="save-experiment-heading" htmlFor="experiment-label" className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Snapshot label</label><input id="experiment-label" value={label} onChange={(event) => setLabel(event.target.value)} placeholder="Hybrid baseline" className="mt-2 h-10 w-full rounded-md border border-white/15 bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ember" /></div>
          <label className="min-w-[170px]"><span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Caption style</span><select value={style} onChange={(event) => setStyle(event.target.value)} className="mt-2 h-10 w-full rounded-md border border-white/15 bg-background px-3 text-sm"><option value="formal">Formal</option><option value="sarcastic">Sarcastic</option><option value="humorous-tech">Humorous Tech</option><option value="humorous-non-tech">Humorous Non-Tech</option></select></label>
          <Button onClick={saveExperiment} disabled={busy || !hasCaptions} className="min-h-11 gap-2"><Save className="h-4 w-4" /> Save Experiment</Button>
        </div>
      </section>

      {run.experiments.length < 2 ? (
        <div className="glass-panel rounded-xl border-dashed p-10 text-center"><Beaker className="mx-auto h-8 w-8 text-muted-foreground" /><div className="mt-3 text-lg font-semibold">Save one more snapshot to compare.</div><p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">Generate captions, save Experiment A, change a supported setting, regenerate the affected stages, then save Experiment B.</p></div>
      ) : (
        <>
          <div className="mb-6 grid gap-3 sm:grid-cols-2"><label><span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Experiment A</span><select value={leftId} onChange={(event) => setLeftId(event.target.value)} className="mt-2 h-10 w-full rounded-md border border-white/15 bg-background px-3 text-sm">{run.experiments.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label><span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Experiment B</span><select value={rightId} onChange={(event) => setRightId(event.target.value)} className="mt-2 h-10 w-full rounded-md border border-white/15 bg-background px-3 text-sm">{run.experiments.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label></div>
          {comparison ? <ComparisonView comparison={comparison} /> : <StaleStageNotice message="Choose two different snapshots to load the comparison." />}
        </>
      )}
      <PrevNext runId={runId} prev={{ to: "/lab/$runId/captions", label: "Captions" }} />
    </div>
  );
}

function ComparisonView({ comparison }: { comparison: ExperimentComparison }) {
  const fields: Array<[string, keyof Experiment | string]> = [["frame method", "frameMethod"], ["frame count", "frameCount"], ["audio mode", "audioMode"], ["audio status", "audioStatus"], ["evidence route", "evidenceRoute"], ["provider", "evidenceProvider"], ["model", "evidenceModel"], ["modality", "evidenceModality"], ["evidence temperature", "evidenceTemperature"], ["caption temperature", "captionTemperature"], ["runtime", "runtimeMs"], ["outcome", "generationOutcome"]];
  return <div className="space-y-6"><div className="grid gap-6 lg:grid-cols-2"><ExperimentColumn side="A" experiment={comparison.left} /><ExperimentColumn side="B" experiment={comparison.right} /></div><div className="overflow-hidden rounded-xl border border-border bg-card"><div className="flex items-center justify-between border-b border-border px-4 py-3"><div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">configuration diff</div><ArrowLeftRight className="h-4 w-4 text-muted-foreground" /></div><table className="w-full text-sm"><tbody className="divide-y divide-border">{fields.map(([label, field]) => <Row key={label} label={label} left={comparison.left[field as keyof Experiment]} right={comparison.right[field as keyof Experiment]} />)}</tbody></table></div></div>;
}

function Row({ label, left, right }: { label: string; left: unknown; right: unknown }) {
  const differs = String(left) !== String(right);
  return <tr><td className="px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{label}</td><td className={cn("px-4 py-2.5 font-mono text-xs", differs && "text-ember")}>{String(left ?? "not recorded")}</td><td className={cn("px-4 py-2.5 font-mono text-xs", differs && "text-ember")}>{String(right ?? "not recorded")}</td></tr>;
}

function ExperimentColumn({ side, experiment }: { side: "A" | "B"; experiment: Experiment }) {
  return <div className={cn("flex flex-col overflow-hidden rounded-xl border border-border bg-card", side === "B" && "bg-ember-soft/25")}><div className="flex items-center justify-between border-b border-border/60 px-5 py-3"><div><div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">experiment {side}</div><div className="mt-0.5 font-display text-2xl">{experiment.label}</div></div><span className="rounded-full border border-border bg-background px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{experiment.style}</span></div><div className="px-5 py-5"><div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">caption</div><p className="mt-2 text-pretty text-[15px] leading-relaxed">{experiment.caption}</p></div></div>;
}
