import { createFileRoute } from "@tanstack/react-router";
import { PrevNext } from "@/components/PrevNext";
import { useRun } from "@/lib/hooks";
import { StageHeader } from "./lab.$runId.video";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Experiment } from "@/lib/types";
import { Beaker, ArrowLeftRight } from "lucide-react";
import { useState } from "react";

export const Route = createFileRoute("/lab/$runId/compare")({
  component: CompareStage,
});

const PRESETS = [
  { id: "methods", label: "Compare frame methods" },
  { id: "counts", label: "Compare frame counts" },
  { id: "temps", label: "Compare temperatures" },
  { id: "audio", label: "Compare visual-only and audio-visual" },
] as const;

function CompareStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const [active, setActive] = useState<(typeof PRESETS)[number]["id"]>("methods");
  if (!run) return null;

  const [a, b] = run.experiments;

  return (
    <div>
      <StageHeader
        eyebrow="stage 06 · compare"
        title="Change one thing. See what breaks."
        description="Run two configurations side by side. Small changes to frames, temperature, or audio produce visibly different captions."
      />

      <div className="mb-6 flex flex-wrap items-center gap-2">
        {PRESETS.map((p) => (
          <Button
            key={p.id}
            variant={active === p.id ? "default" : "outline"}
            size="sm"
            className="gap-2"
            onClick={() => setActive(p.id)}
          >
            <Beaker className="h-3.5 w-3.5" /> {p.label}
          </Button>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ExperimentColumn side="A" experiment={a} />
        <ExperimentColumn side="B" experiment={b} />
      </div>

      <div className="mt-6 overflow-hidden rounded-xl border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            configuration diff
          </div>
          <ArrowLeftRight className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-background/40 text-left font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                <th className="px-4 py-2 font-normal">metric</th>
                <th className="px-4 py-2 font-normal">A · {a.label}</th>
                <th className="px-4 py-2 font-normal">B · {b.label}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              <Row k="extraction method" a={a.frameMethod} b={b.frameMethod} />
              <Row k="frame count" a={a.frameCount} b={b.frameCount} />
              <Row
                k="selected timestamps"
                a={a.timestamps.map((t) => `${t}s`).join(", ")}
                b={b.timestamps.map((t) => `${t}s`).join(", ")}
              />
              <Row k="audio mode" a={a.audioMode} b={b.audioMode} />
              <Row k="evidence model" a={a.evidenceModel} b={b.evidenceModel} />
              <Row k="caption temperature" a={a.captionTemperature} b={b.captionTemperature} />
              <Row k="runtime" a={`${(a.runtimeMs / 1000).toFixed(2)}s`} b={`${(b.runtimeMs / 1000).toFixed(2)}s`} />
            </tbody>
          </table>
        </div>
      </div>

      <PrevNext runId={runId} prev={{ to: "/lab/$runId/captions", label: "Captions" }} />
    </div>
  );
}

function Row({ k, a, b }: { k: string; a: React.ReactNode; b: React.ReactNode }) {
  const differs = String(a) !== String(b);
  return (
    <tr>
      <td className="px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{k}</td>
      <td className={cn("px-4 py-2.5 font-mono text-xs", differs && "text-ember")}>{a}</td>
      <td className={cn("px-4 py-2.5 font-mono text-xs", differs && "text-ember")}>{b}</td>
    </tr>
  );
}

function ExperimentColumn({ side, experiment }: { side: "A" | "B"; experiment: Experiment }) {
  return (
    <div
      className={cn(
        "flex flex-col overflow-hidden rounded-xl border border-border bg-card",
        side === "B" && "bg-ember-soft/25",
      )}
    >
      <div className="flex items-center justify-between border-b border-border/60 px-5 py-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            experiment {side}
          </div>
          <div className="mt-0.5 font-display text-2xl">{experiment.label}</div>
        </div>
        <span className="rounded-full border border-border bg-background px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {experiment.style}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-px bg-border">
        <Mini label="frames" value={experiment.frameCount} />
        <Mini label="temp" value={experiment.captionTemperature.toFixed(1)} />
        <Mini label="runtime" value={`${(experiment.runtimeMs / 1000).toFixed(1)}s`} />
      </div>
      <div className="flex-1 px-5 py-5">
        <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          caption
        </div>
        <p className="mt-2 text-pretty text-[15px] leading-relaxed">{experiment.caption}</p>
      </div>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="bg-card px-4 py-3">
      <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 font-mono text-sm">{value}</div>
    </div>
  );
}
