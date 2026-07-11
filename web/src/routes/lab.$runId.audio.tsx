import { createFileRoute } from "@tanstack/react-router";
import { ConfigSection, Field } from "@/components/ConfigSection";
import { PrevNext } from "@/components/PrevNext";
import { AudioWaveform } from "@/components/AudioWaveform";
import { useRun } from "@/lib/hooks";
import { StageHeader } from "./lab.$runId.video";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Play, RotateCw, Volume2, VolumeX, Info } from "lucide-react";
import { useState } from "react";
import { api } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { runKey } from "@/lib/hooks";
import type { AudioMode, AudioConfig } from "@/lib/types";
import { ProcessingState } from "@/components/StateViews";

export const Route = createFileRoute("/lab/$runId/audio")({
  component: AudioStage,
});

const MODES: { id: AudioMode; label: string; note: string }[] = [
  { id: "disabled", label: "Disabled", note: "Skip audio entirely, force the visual-only route." },
  { id: "automatic", label: "Automatic", note: "Analyze audio if a usable non-silent window exists." },
  { id: "always", label: "Always analyze", note: "Always include an audio segment even if quiet." },
];

function AudioStage() {
  const { runId } = Route.useParams();
  const { data: run } = useRun(runId);
  const qc = useQueryClient();
  const [mode, setMode] = useState<AudioMode>(run?.audio.config.mode ?? "automatic");
  const [maxDur, setMaxDur] = useState(run?.audio.config.maxDurationSec ?? 8);
  const [rate, setRate] = useState(String(run?.audio.config.sampleRateHz ?? 16000));
  const [minRms, setMinRms] = useState(run?.audio.config.minRmsEnergy ?? 0.02);
  const [strategy, setStrategy] = useState<AudioConfig["strategy"]>(
    run?.audio.config.strategy ?? "highest-energy",
  );
  const [busy, setBusy] = useState(false);

  if (!run || run.stages.audio !== "complete") return <ProcessingState />;

  async function apply() {
    setBusy(true);
    const updated = await api.postAudio(runId, {
      mode,
      maxDurationSec: maxDur,
      sampleRateHz: Number(rate),
      minRmsEnergy: minRms,
      strategy,
    });
    qc.setQueryData(runKey(runId), updated);
    setBusy(false);
  }

  return (
    <div>
      <StageHeader
        eyebrow="stage 03 · audio"
        title="Find a useful listening window"
        description="Audio energy helps select a candidate segment. Gemma decides whether it actually contains speech or useful evidence."
      />

      <div className="mb-6 flex items-start gap-3 rounded-xl border border-lab/30 bg-lab-soft/50 p-4 text-sm">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-lab" />
        <p className="text-pretty">
          <strong>Audio energy helps select a useful segment.</strong> It does not prove that
          speech is present. Gemma determines whether speech or useful audio evidence exists.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
        <ConfigSection
          title="Audio configuration"
          actions={
            <Button size="sm" className="gap-1.5" onClick={apply} disabled title="Interactive reruns coming in the next integration phase">
              {busy ? <RotateCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              Apply
            </Button>
          }
        >
          <RadioGroup value={mode} onValueChange={(v) => setMode(v as AudioMode)} className="grid gap-2">
            {MODES.map((m) => (
              <label
                key={m.id}
                htmlFor={`am-${m.id}`}
                className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition ${
                  mode === m.id
                    ? "border-ember bg-ember-soft"
                    : "border-white/10 bg-background hover:border-white/20"
                }`}
              >
                <RadioGroupItem id={`am-${m.id}`} value={m.id} className="mt-0.5" />
                <div>
                  <Label htmlFor={`am-${m.id}`} className="cursor-pointer font-medium">{m.label}</Label>
                  <p className="mt-0.5 text-xs text-muted-foreground">{m.note}</p>
                </div>
              </label>
            ))}
          </RadioGroup>

          <Field label="Max analysis duration" hint={<span className="font-mono">{maxDur}s</span>}>
            <Slider value={[maxDur]} onValueChange={([v]) => setMaxDur(v)} min={1} max={30} step={1} />
          </Field>
          <Field label="Sample rate">
            <Select value={rate} onValueChange={setRate}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="8000">8 kHz</SelectItem>
                <SelectItem value="16000">16 kHz</SelectItem>
                <SelectItem value="22050">22.05 kHz</SelectItem>
                <SelectItem value="44100">44.1 kHz</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field label="Min RMS energy" hint={<span className="font-mono">{minRms.toFixed(3)}</span>}>
            <Slider value={[minRms]} onValueChange={([v]) => setMinRms(v)} min={0.001} max={0.2} step={0.001} />
          </Field>
          <Field label="Selection strategy">
            <Select value={strategy} onValueChange={(v) => setStrategy(v as AudioConfig["strategy"])}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="highest-energy">Highest energy window</SelectItem>
                <SelectItem value="first-non-silent">First non-silent window</SelectItem>
                <SelectItem value="custom-range">Custom range</SelectItem>
              </SelectContent>
            </Select>
          </Field>
        </ConfigSection>

        <div className="space-y-4">
          {run.audio.segment.artifactAvailable ? <AudioWaveform segment={run.audio.segment} durationSec={run.video.durationSec} /> : <div className="rounded-xl border border-dashed border-border bg-card p-6 text-sm text-muted-foreground">The pipeline securely removed its temporary audio window, so no waveform or playable audio artifact is available for this run.</div>}

          <div className="grid gap-3 sm:grid-cols-2">
            <StatusCard
              icon={run.audio.segment.hasAudioStream ? Volume2 : VolumeX}
              label="Audio stream"
              value={run.audio.segment.hasAudioStream ? "present" : "missing"}
              tone={run.audio.segment.hasAudioStream ? "ok" : "warn"}
            />
            <StatusCard
              icon={Info}
              label="Energy candidate"
              value={run.audio.segment.energyCandidateFound ? "found" : "none"}
              tone={run.audio.segment.energyCandidateFound ? "ok" : "warn"}
            />
          </div>

          <div className="rounded-xl border border-border bg-card p-4">
            <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              route explanation
            </div>
            <p className="mt-2 text-sm">{run.audio.segment.routeExplanation}</p>
          </div>

          <div className="flex items-center gap-3 rounded-xl border border-border bg-card p-4">
            <button disabled aria-label="Audio artifact unavailable" className="flex h-11 w-11 items-center justify-center rounded-full bg-muted text-muted-foreground">
              <Play className="ml-0.5 h-4 w-4" fill="currentColor" />
            </button>
            <div className="flex-1">
              <div className="font-mono text-xs text-muted-foreground">selected segment</div>
              <div className="font-mono text-sm">
                {run.audio.segment.startSec.toFixed(2)}s → {run.audio.segment.endSec.toFixed(2)}s
              </div>
            </div>
            <div className="font-mono text-xs text-muted-foreground">
              rms {run.audio.segment.rms.toFixed(3)}
            </div>
          </div>
        </div>
      </div>

      <PrevNext
        runId={runId}
        prev={{ to: "/lab/$runId/frames", label: "Frames" }}
        next={{ to: "/lab/$runId/evidence", label: "Evidence" }}
      />
    </div>
  );
}

function StatusCard({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  tone: "ok" | "warn";
}) {
  return (
    <div className={`rounded-xl border p-4 ${tone === "ok" ? "border-border bg-card" : "border-ember/30 bg-ember-soft/40"}`}>
      <div className="flex items-center gap-2">
        <Icon className={`h-4 w-4 ${tone === "ok" ? "text-success" : "text-warning"}`} />
        <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          {label}
        </span>
      </div>
      <div className="mt-1 font-mono text-sm capitalize">{value}</div>
    </div>
  );
}
