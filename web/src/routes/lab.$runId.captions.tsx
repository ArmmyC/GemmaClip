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
import { useState } from "react";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { CaptionStyle, CaptionConfig } from "@/lib/types";
import { cn } from "@/lib/utils";
import { GenerationOutcomeNotice } from "@/components/GenerationOutcomeNotice";

export const Route = createFileRoute("/lab/$runId/captions")({
  component: CaptionsStage,
});

const STYLES: { id: CaptionStyle; label: string }[] = [
  { id: "formal", label: "Formal" },
  { id: "sarcastic", label: "Sarcastic" },
  { id: "humorous-tech", label: "Humorous · Tech" },
  { id: "humorous-non-tech", label: "Humorous · Non-Tech" },
  { id: "social", label: "Social Media" },
  { id: "accessibility", label: "Accessibility" },
];

function CaptionsStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [temp, setTemp] = useState(run?.captions.config.temperature ?? 0.7);
  const [minW, setMinW] = useState(run?.captions.config.minWords ?? 12);
  const [maxW, setMaxW] = useState(run?.captions.config.maxWords ?? 48);
  const [strict, setStrict] = useState(run?.captions.config.strictGrounding ?? true);
  const [audioMode, setAudioMode] = useState<CaptionConfig["audioEvidenceMode"]>(
    run?.captions.config.audioEvidenceMode ?? "use-if-present",
  );
  const [repair, setRepair] = useState(run?.captions.config.focusedRepair ?? true);
  const [styles, setStyles] = useState<CaptionStyle[]>(
    run?.captions.config.styles ?? ["formal", "humorous-tech", "social", "accessibility"],
  );
  const [busy, setBusy] = useState(false);

  if (!run) return null;

  function toggleStyle(s: CaptionStyle) {
    setStyles((cur) => (cur.includes(s) ? cur.filter((x) => x !== s) : [...cur, s]));
  }

  async function generate() {
    setBusy(true);
    const updated = await api.postCaptions(runId, {
      model: "gemma-4-31b",
      temperature: temp,
      minWords: minW,
      maxWords: maxW,
      strictGrounding: strict,
      audioEvidenceMode: audioMode,
      focusedRepair: repair,
      styles,
    });
    qc.setQueryData(runKey(runId), updated);
    setBusy(false);
  }

  return (
    <div>
      <StageHeader
        eyebrow="stage 05 · captions"
        title="Write, styled and grounded"
        description="Captions can only reference facts already established as evidence. Focused repair rewrites lines that stray."
      />
      <GenerationOutcomeNotice outcome={run.generationOutcome} compact />

      <div className="grid gap-6 lg:grid-cols-[1fr_1.6fr]">
        <ConfigSection
          title="Caption config"
          description={
            <>
              Caption model: <span className="font-mono text-foreground">Gemma 4 · 31B</span>
            </>
          }
          actions={
            <Button size="sm" className="gap-1.5" onClick={generate} disabled title="Interactive reruns coming in the next integration phase">
              {busy ? <RotateCw className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              Generate
            </Button>
          }
        >
          <Field label="Temperature" hint={<span className="font-mono">{temp.toFixed(2)}</span>}>
            <Slider value={[temp]} onValueChange={([v]) => setTemp(v)} min={0} max={1.2} step={0.05} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Min words" hint={<span className="font-mono">{minW}</span>}>
              <Slider value={[minW]} onValueChange={([v]) => setMinW(v)} min={4} max={40} step={1} />
            </Field>
            <Field label="Max words" hint={<span className="font-mono">{maxW}</span>}>
              <Slider value={[maxW]} onValueChange={([v]) => setMaxW(v)} min={12} max={120} step={1} />
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
          <label className="flex items-center justify-between rounded-lg border border-border bg-background p-3">
            <div>
              <div className="text-sm font-medium">Strict grounding</div>
              <div className="text-xs text-muted-foreground">Reject captions that add unsupported claims.</div>
            </div>
            <Switch checked={strict} onCheckedChange={setStrict} />
          </label>
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
                      ? "border-ink bg-ink text-paper"
                      : "border-border bg-background text-muted-foreground hover:border-ink/40",
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
