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
import { useState } from "react";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { ModelRoute } from "@/lib/types";

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

  if (!run) return null;
  const ev = run.evidence.result;

  async function apply() {
    setBusy(true);
    const updated = await api.postEvidence(runId, {
      route,
      temperature: temp,
      maxTokens,
      provider: run!.evidence.config.provider,
      showPromptStructure: showPrompt,
      showRawJson: showRaw,
    });
    qc.setQueryData(runKey(runId), updated);
    setBusy(false);
  }

  return (
    <div>
      <StageHeader
        eyebrow="stage 04 · evidence"
        title="Ground the model in what's actually there"
        description="Structured evidence separates what Gemma verified from what it might have misread. Captions can only reference this object."
      />

      <div className="mb-6">
        <RouteDecision
          selected={ev.selectedRoute}
          reason={ev.routeReason}
          auto={route === "auto"}
          audio={ev.audio}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.7fr]">
        <ConfigSection
          title="Evidence config"
          description="No secrets, API keys, or hidden chain-of-thought are ever surfaced here."
          actions={
            <Button size="sm" className="gap-1.5" onClick={apply} disabled={busy}>
              {busy ? <RotateCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              Run evidence
            </Button>
          }
        >
          <RadioGroup value={route} onValueChange={(v) => setRoute(v as ModelRoute)} className="grid gap-2">
            {ROUTES.map((r) => (
              <label
                key={r.id}
                htmlFor={`r-${r.id}`}
                className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition ${
                  route === r.id
                    ? "border-ink bg-accent"
                    : "border-border bg-background hover:border-ink/40"
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
            <code className="rounded bg-muted px-2 py-1 font-mono text-xs text-muted-foreground">
              provider://gemma-gateway
            </code>
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
            <EvidenceCard label="verified description" className="md:col-span-2">
              {ev.verifiedDescription}
            </EvidenceCard>
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
                <div>language: {ev.audio.language ?? "—"}</div>
                <div>consistency: {ev.audio.visualConsistency}</div>
                <div>transcript: {ev.audio.transcript ?? "—"}</div>
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
