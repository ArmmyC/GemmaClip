import { createFileRoute } from "@tanstack/react-router";
import { ConfigSection, Field } from "@/components/ConfigSection";
import { PrevNext } from "@/components/PrevNext";
import { RouteDecision } from "@/components/RouteDecision";
import { EvidenceCard, EvidenceList } from "@/components/EvidenceCard";
import { RawJsonViewer } from "@/components/RawJsonViewer";
import { useRun } from "@/lib/hooks";
import { StageHeader } from "./lab.$runId.video";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Play, RotateCw } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { EvidenceRequest, ModelRoute } from "@/lib/types";
import { InvalidationPreview, ProcessingState, StageErrorState, StaleStageNotice } from "@/components/StateViews";

export const Route = createFileRoute("/lab/$runId/evidence")({
  component: EvidenceStage,
});

const ROUTES: { id: ModelRoute; label: string; note: string }[] = [
  { id: "auto", label: "Automatic routing", note: "Route by audio availability and budget." },
  { id: "gemma-4-26b-a4b", label: "Gemma 4 · 26B A4B", note: "Visual evidence only." },
  { id: "gemma-4-12b-unified", label: "Gemma 4 · 12B Unified", note: "Visual and audio evidence." },
];

function EvidenceStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [route, setRoute] = useState<ModelRoute>(run?.evidence.config.route ?? "auto");
  const [temp, setTemp] = useState(run?.evidence.config.temperature ?? 0.2);
  const [maxTokens, setMaxTokens] = useState(run?.evidence.config.maxTokens ?? 1200);
  const [showPrompt, setShowPrompt] = useState(run?.evidence.config.showPromptStructure ?? false);
  const [showRaw, setShowRaw] = useState(run?.evidence.config.showRawJson ?? false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    if (!run) return;
    setRoute(run.evidence.config.route); setTemp(run.evidence.config.temperature); setMaxTokens(run.evidence.config.maxTokens);
    setShowPrompt(run.evidence.config.showPromptStructure); setShowRaw(run.evidence.config.showRawJson);
  }, [run]);

  if (!run) return <ProcessingState />;
  if (run.stages.evidence === "active") return <ProcessingState description={run.progressMessage ?? "Building grounded evidence."} />;
  if (run.stages.evidence === "error") return <StageErrorState stage="Evidence" description={run.stageErrors?.evidence ?? run.error ?? undefined} onRetry={() => apply()} />;
  const ev = {
    ...run.evidence.result,
    scene: run.evidence.result.scene ?? "",
    subjects: run.evidence.result.subjects ?? [],
    actions: run.evidence.result.actions ?? [],
    setting: run.evidence.result.setting ?? "",
    visibleObjects: run.evidence.result.visibleObjects ?? [],
    mood: run.evidence.result.mood ?? "",
    cameraNotes: run.evidence.result.cameraNotes ?? "",
    temporalProgression: run.evidence.result.temporalProgression ?? [],
    verifiedDescription: run.evidence.result.verifiedDescription ?? "",
    possibleMisreads: run.evidence.result.possibleMisreads ?? [],
    unsupportedClaims: run.evidence.result.unsupportedClaims ?? [],
    styleHooks: run.evidence.result.styleHooks ?? [],
    audio: run.evidence.result.audio ?? { status: "unavailable" as const, speechPresent: false, language: null, transcript: null, visualConsistency: "unknown" as const, captionSafeFacts: [] },
  };
  const hasEvidence = Boolean(run.evidence.result.audio);
  const draft: EvidenceRequest = { route, temperature: temp, maxTokens, provider: run.evidence.config.provider, showPromptStructure: showPrompt, showRawJson: showRaw };
  const dirty = route !== run.evidence.config.route || temp !== run.evidence.config.temperature || maxTokens !== run.evidence.config.maxTokens;

  async function apply() {
    setBusy(true);
    setError(null); setNotice(null);
    try {
      const updated = await api.postEvidence(runId, draft);
      qc.setQueryData(runKey(runId), updated);
      setNotice("Evidence updated. Captions and Compare require regeneration.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Evidence generation failed safely.");
    } finally { setBusy(false); }
  }

  return (
    <div>
      <StageHeader
        eyebrow="stage 04 · evidence"
        title="Ground the model in what's actually there"
        description="Structured evidence separates what Gemma verified from what it might have misread. Captions can only reference this object."
      />
      {run.stages.evidence === "invalidated" && <StaleStageNotice />}
      {dirty && <InvalidationPreview stage="evidence" />}
      {notice && <div className="mb-4 rounded-lg border border-success/30 bg-success/5 px-3 py-2 text-sm text-success" role="status">{notice}</div>}
      {error && <div className="mb-4 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-danger" role="alert">{error}</div>}

      <div className="mb-6">
        {hasEvidence ? <RouteDecision
          selected={ev.selectedRoute}
          reason={ev.routeReason}
          auto={route === "auto"}
           audio={ev.audio}
           provider={ev.routeProvider}
           model={ev.routeModel}
           modality={ev.routeModality}
          audioFallbackOccurred={ev.audioFallbackOccurred}
        /> : <div className="rounded-lg border border-white/10 bg-card/50 px-4 py-3 text-sm text-muted-foreground">Evidence has not been built for this run yet. Configure the route and build evidence to see provenance.</div>}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.7fr]">
        <ConfigSection
          title="Evidence config"
          description="Choose the Gemma evidence route. No secrets, API keys, or hidden chain-of-thought are surfaced."
          actions={
            <div className="flex flex-wrap gap-2">
              <Button variant="ghost" size="sm" onClick={() => { setRoute(run.evidence.config.route); setTemp(run.evidence.config.temperature); setMaxTokens(run.evidence.config.maxTokens); }} disabled={busy || !dirty}>Reset</Button>
              <Button size="sm" className="gap-1.5" onClick={apply} disabled={busy || run.stages.frames !== "complete" || (!dirty && run.stages.evidence === "complete")}>
              {busy ? <RotateCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              Build Evidence
              </Button>
            </div>
          }
        >
          <p className="rounded-md border border-white/10 bg-background/60 px-3 py-2 text-xs text-muted-foreground">
            {dirty ? "Unsaved configuration. Captions and Compare will be invalidated." : "Stored configuration. Adjust the route or temperature, then build evidence."}
          </p>
          <RadioGroup value={route} onValueChange={(v) => setRoute(v as ModelRoute)} className="grid gap-2">
            {ROUTES.map((r) => (
              <label
                key={r.id}
                htmlFor={`r-${r.id}`}
                className={`flex items-start gap-3 rounded-lg border p-3 transition ${
                  route === r.id
                    ? "border-ember bg-ember-soft"
                    : "border-white/10 bg-background hover:border-white/20"
                }`}
              >
                <RadioGroupItem id={`r-${r.id}`} value={r.id} className="mt-0.5" />
                <div>
                  <Label htmlFor={`r-${r.id}`} className="cursor-pointer font-medium">
                    {r.label}
                  </Label>
                  <p className="mt-0.5 text-xs text-muted-foreground">{r.note}</p>
                </div>
              </label>
            ))}
          </RadioGroup>

          <Field label="Evidence temperature" hint={<span className="font-mono">{temp.toFixed(2)}</span>}>
            <Slider value={[temp]} onValueChange={([v]) => setTemp(v)} min={0} max={1} step={0.05} />
          </Field>
          <Field label="Max tokens" hint={<span className="font-mono">{maxTokens}</span>}>
            <Slider value={[maxTokens]} onValueChange={([v]) => setMaxTokens(v)} min={256} max={4096} step={64} />
          </Field>
          <Field label="Provider">
            <span className="font-mono text-xs text-muted-foreground">Automatic routed provider (server configured)</span>
          </Field>
          <label className="flex items-center justify-between rounded-lg border border-border bg-background p-3">
            <div>
              <div className="text-sm font-medium">Show prompt structure</div>
              <div className="text-xs text-muted-foreground">Reveal the message layout, not the system content.</div>
            </div>
            <Switch checked={showPrompt} onCheckedChange={setShowPrompt} />
          </label>
          <label className="flex items-center justify-between rounded-lg border border-border bg-background p-3">
            <div>
              <div className="text-sm font-medium">Show raw JSON</div>
              <div className="text-xs text-muted-foreground">Inspect the full evidence object.</div>
            </div>
            <Switch checked={showRaw} onCheckedChange={setShowRaw} />
          </label>
        </ConfigSection>

        <div className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="md:col-span-2 flex items-center gap-3 border-b border-white/10 pb-2 pt-2 font-mono text-[10px] uppercase tracking-[0.2em] text-success"><span className="h-px w-5 bg-success" /> Verified observations</div>
            <EvidenceCard label="scene">{ev.scene}</EvidenceCard>
            <EvidenceCard label="setting">{ev.setting}</EvidenceCard>
            <EvidenceCard label="subjects">
              <EvidenceList items={ev.subjects} />
            </EvidenceCard>
            <EvidenceCard label="actions">
              <EvidenceList items={ev.actions} />
            </EvidenceCard>
            <EvidenceCard label="visible objects">
              <EvidenceList items={ev.visibleObjects} />
            </EvidenceCard>
            <EvidenceCard label="mood">{ev.mood}</EvidenceCard>
            <EvidenceCard label="camera notes">{ev.cameraNotes}</EvidenceCard>
            <EvidenceCard label="temporal progression">
              <EvidenceList items={ev.temporalProgression} />
            </EvidenceCard>
            <EvidenceCard label="verified description" className="border-ember/30 bg-ember-soft/5 md:col-span-2">
              {ev.verifiedDescription}
            </EvidenceCard>
            <div className="md:col-span-2 flex items-center gap-3 border-b border-white/10 pb-2 pt-6 font-mono text-[10px] uppercase tracking-[0.2em] text-warning"><span className="h-px w-5 bg-warning" /> Do not claim</div>
            <EvidenceCard label="possible misreads" tone="warn">
              <EvidenceList items={ev.possibleMisreads} />
            </EvidenceCard>
            <EvidenceCard label="unsupported claims" tone="warn">
              <EvidenceList items={ev.unsupportedClaims} />
            </EvidenceCard>
            <EvidenceCard label="style hooks" tone="info">
              <EvidenceList items={ev.styleHooks} />
            </EvidenceCard>
            <EvidenceCard label="audio · status" tone="info">
              <div className="space-y-1 font-mono text-xs">
                <div>status: {ev.audio.status}</div>
                <div>speech: {ev.audio.speechPresent ? "present" : "not detected"}</div>
                <div>language: {ev.audio.language ?? "none recorded"}</div>
                <div>consistency: {ev.audio.visualConsistency}</div>
                <div>transcript: {ev.audio.transcript ?? "none recorded"}</div>
              </div>
            </EvidenceCard>
            <EvidenceCard label="caption-safe audio facts" tone="info" className="md:col-span-2">
              <EvidenceList items={ev.audio.captionSafeFacts} />
            </EvidenceCard>
          </div>

          <RawJsonViewer data={ev} defaultOpen={showRaw} />
        </div>
      </div>

      <PrevNext
        runId={runId}
        prev={{ to: "/lab/$runId/audio", label: "Audio" }}
        next={{ to: "/lab/$runId/captions", label: "Captions" }}
      />
    </div>
  );
}
