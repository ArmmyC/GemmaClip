import { createFileRoute } from "@tanstack/react-router";
import { ConfigSection, Field } from "@/components/ConfigSection";
import { PrevNext } from "@/components/PrevNext";
import { CaptionCard } from "@/components/CaptionCard";
import { useRun } from "@/lib/hooks";
import { StageHeader } from "./lab.$runId.video";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Sparkles, RotateCw } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { CaptionConfig, CaptionRequest, CaptionStyle } from "@/lib/types";
import { cn } from "@/lib/utils";
import { GenerationOutcomeNotice } from "@/components/GenerationOutcomeNotice";
import { ProcessingState, StageErrorState, StaleStageNotice } from "@/components/StateViews";

export const Route = createFileRoute("/lab/$runId/captions")({
  component: CaptionsStage,
});

const STYLES: { id: CaptionStyle; label: string }[] = [
  { id: "formal", label: "Formal" },
  { id: "sarcastic", label: "Sarcastic" },
  { id: "humorous-tech", label: "Humorous · Tech" },
  { id: "humorous-non-tech", label: "Humorous · Non-Tech" },
];

function CaptionsStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [temp, setTemp] = useState(run?.captions.config.temperature ?? 0.7);
  const [minW, setMinW] = useState(run?.captions.config.minWords ?? 12);
  const [maxW, setMaxW] = useState(run?.captions.config.maxWords ?? 48);
  const strict = true;
  const [audioMode, setAudioMode] = useState<CaptionConfig["audioEvidenceMode"]>(
    run?.captions.config.audioEvidenceMode ?? "use-if-present",
  );
  const [repair, setRepair] = useState(run?.captions.config.focusedRepair ?? true);
  const [styles, setStyles] = useState<CaptionStyle[]>(
    run?.captions.config.styles ?? ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"],
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    if (!run) return;
    setTemp(run.captions.config.temperature); setMinW(run.captions.config.minWords); setMaxW(run.captions.config.maxWords);
    setAudioMode(run.captions.config.audioEvidenceMode); setRepair(run.captions.config.focusedRepair); setStyles(run.captions.config.styles);
  }, [run]);

  if (!run) return <ProcessingState />;
  if (run.stages.captions === "active") return <ProcessingState description={run.progressMessage ?? "Writing captions."} />;
  if (run.stages.captions === "error") return <StageErrorState stage="Captions" description={run.stageErrors?.captions ?? run.error ?? undefined} onRetry={() => generate()} />;
  const draft: CaptionRequest = { temperature: temp, minWords: minW, maxWords: maxW, strictGrounding: strict, audioEvidenceMode: audioMode, focusedRepair: repair, styles };
  const dirty = temp !== run.captions.config.temperature || minW !== run.captions.config.minWords || maxW !== run.captions.config.maxWords || audioMode !== run.captions.config.audioEvidenceMode || repair !== run.captions.config.focusedRepair || JSON.stringify(styles) !== JSON.stringify(run.captions.config.styles);

  function toggleStyle(s: CaptionStyle) {
    setStyles((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));
  }

  async function generate() {
    setBusy(true);
    setError(null); setNotice(null);
    try {
      const updated = await api.postCaptions(runId, draft);
      qc.setQueryData(runKey(runId), updated);
      setNotice("Captions generated. Save this configuration as an experiment when ready.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Caption generation failed safely.");
    } finally { setBusy(false); }
  }

  return (
    <div>
      <StageHeader
        eyebrow="stage 05 · captions"
        title="Write, styled and grounded"
        description="Captions can only reference facts already established as evidence. Focused repair rewrites lines that stray."
      />
      <GenerationOutcomeNotice outcome={run.generationOutcome} compact />
      {run.stages.captions === "invalidated" && <StaleStageNotice />}
      {notice && <div className="mb-4 rounded-lg border border-success/30 bg-success/5 px-3 py-2 text-sm text-success" role="status">{notice}</div>}
      {error && <div className="mb-4 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger" role="alert">{error}</div>}

      <div className="grid gap-6 lg:grid-cols-[1fr_1.6fr]">
        <ConfigSection
          title="Caption config"
          description={
            <>
              Caption model: <span className="font-mono text-foreground">Gemma 4 · 31B</span>
            </>
          }
          actions={
            <Button size="sm" className="gap-1.5" onClick={generate} disabled={busy || run.stages.evidence !== "complete" || (!dirty && run.stages.captions === "complete")}>
              {busy ? <RotateCw className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              Generate
            </Button>
          }
        >
          <div className="flex items-center justify-between gap-3 rounded-md border border-white/10 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
            <span>{dirty ? "Unsaved configuration. Compare will be invalidated after generation." : "Stored configuration. Choose styles and settings, then generate captions."}</span>
            <Button variant="ghost" size="sm" onClick={() => { setTemp(run.captions.config.temperature); setMinW(run.captions.config.minWords); setMaxW(run.captions.config.maxWords); setAudioMode(run.captions.config.audioEvidenceMode); setRepair(run.captions.config.focusedRepair); setStyles(run.captions.config.styles); }} disabled={busy || !dirty}>Reset</Button>
          </div>
          <Field label="Temperature" hint={<span className="font-mono">{temp.toFixed(2)}</span>}>
            <Slider value={[temp]} onValueChange={([v]) => setTemp(v)} min={0} max={1.2} step={0.05} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Min words" hint={<span className="font-mono">{minW}</span>}>
              <Slider value={[minW]} onValueChange={([v]) => setMinW(Math.min(v, maxW))} min={1} max={120} step={1} />
            </Field>
            <Field label="Max words" hint={<span className="font-mono">{maxW}</span>}>
              <Slider value={[maxW]} onValueChange={([v]) => setMaxW(Math.max(v, minW))} min={1} max={120} step={1} />
            </Field>
          </div>
          <Field label="Audio evidence">
            <Select value={audioMode} onValueChange={(v) => setAudioMode(v as CaptionConfig["audioEvidenceMode"])}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ignore">Ignore</SelectItem>
                <SelectItem value="use-if-present">Use if present</SelectItem>
                <SelectItem value="require">Require</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <div className="flex items-center justify-between rounded-lg border border-border bg-background p-3">
            <div>
              <div className="text-sm font-medium">Strict grounding</div>
              <div className="text-xs text-muted-foreground">Always enabled so captions cannot add unsupported claims.</div>
            </div>
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-success">required</span>
          </div>
          <label className="flex items-center justify-between rounded-lg border border-border bg-background p-3">
            <div>
              <div className="text-sm font-medium">Focused repair</div>
              <div className="text-xs text-muted-foreground">Repair problem lines instead of rejecting the whole caption.</div>
            </div>
            <Switch checked={repair} onCheckedChange={setRepair} />
          </label>

          <Field label="Styles">
            <div className="flex flex-wrap gap-1.5">
              {STYLES.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => toggleStyle(s.id)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs transition",
                    styles.includes(s.id)
                      ? "border-foreground bg-foreground text-background"
                      : "border-white/10 bg-background text-muted-foreground hover:border-white/20",
                  )}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </Field>
        </ConfigSection>

        <div className="grid gap-4 md:grid-cols-2">
          {run.captions.results.map((c) => (
            <CaptionCard key={c.id} caption={c} />
          ))}
        </div>
      </div>

      <PrevNext
        runId={runId}
        prev={{ to: "/lab/$runId/evidence", label: "Evidence" }}
        next={{ to: "/lab/$runId/compare", label: "Compare" }}
      />
    </div>
  );
}
