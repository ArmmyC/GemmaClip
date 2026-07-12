import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowRight, AudioLines, Braces, Clapperboard, Layers3 } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Button } from "@/components/ui/button";
import { startQuickUpload } from "@/lib/quick-flow";
import { createAndProbeManualRun } from "@/lib/api";
import { ServiceHealthNotice } from "@/components/ServiceHealth";
import { useHealth } from "@/lib/hooks";
import { getHealthActionState } from "@/lib/health";

export const Route = createFileRoute("/")({ component: Landing });

const PIPELINE = [
  { icon: Layers3, label: "Frame selection", note: "Anchor and high-change moments" },
  { icon: AudioLines, label: "Audio check", note: "Optional bounded evidence window" },
  { icon: Braces, label: "Grounded evidence", note: "Route and structured facts" },
  { icon: Clapperboard, label: "Caption writing", note: "Four grounded styles" },
];

function Landing() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const health = useHealth();
  const { serviceUnavailable, generationUnavailable } = getHealthActionState(health);

  async function generate() {
    if (!file) return;
    setBusy(true); setError(null);
    try {
      await startQuickUpload(file, (id) => navigate({ to: "/quick", search: { runId: id } as never }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Upload failed.");
    } finally { setBusy(false); }
  }

  async function openLab() {
    if (!file) return;
    setBusy(true); setError(null);
    try {
      const run = await createAndProbeManualRun(file, "balanced");
      navigate({ to: "/lab/$runId/video", params: { runId: run.id } });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The Lab run could not be created.");
    } finally { setBusy(false); }
  }

  return (
    <div className="min-h-[100dvh] bg-background text-foreground">
      <AppHeader variant="landing" />
      <main>
        <div className="mx-auto max-w-[1440px] px-6 pt-4 lg:px-10">
          <ServiceHealthNotice />
        </div>
        <section className="relative overflow-hidden signal-glow">
          <div className="pointer-events-none absolute inset-0 signal-grid opacity-80" />
          <div className="relative mx-auto grid max-w-[1440px] gap-12 px-6 pb-20 pt-14 lg:grid-cols-[1.05fr_0.95fr] lg:items-end lg:px-10 lg:pb-28 lg:pt-20">
            <div className="max-w-3xl">
              <div className="mb-7 flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                <span className="h-px w-8 bg-ember" aria-hidden="true" />
                video captioning / pure Gemma
              </div>
              <h1 className="max-w-3xl font-display text-5xl font-semibold leading-[0.96] tracking-[-0.055em] text-balance sm:text-7xl lg:text-8xl">
                Build grounded captions.<br />
                <span className="text-muted-foreground">Inspect every decision.</span>
              </h1>
              <p className="mt-7 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
                Drop a video. GemmaClip selects important moments, checks useful audio, builds structured evidence, and writes captions with Gemma.
              </p>
            </div>

            <div className="glass-panel rounded-2xl p-2 lg:translate-y-2">
              <UploadDropzone onFile={setFile} disabled={Boolean(serviceUnavailable)} />
              <div className="flex flex-wrap gap-3 px-4 pb-4 pt-4 sm:px-6">
                <Button size="lg" className="min-h-11 gap-2 rounded-lg px-5" onClick={generate} disabled={!file || busy || generationUnavailable} title={generationUnavailable ? "Gemma generation is not configured or the service is unavailable." : undefined}>
                  {busy ? "Starting run" : "Generate captions"} <ArrowRight className="h-4 w-4" />
                </Button>
                <Button size="lg" variant="outline" className="min-h-11 gap-2 rounded-lg border-white/15 px-5" onClick={openLab} disabled={!file || busy || Boolean(serviceUnavailable)}>
                  Open in Gemma Lab
                </Button>
              </div>
              {error && <p role="alert" className="px-4 pb-4 text-sm text-danger sm:px-6">{error}</p>}
            </div>
          </div>
        </section>

        <section className="border-y border-white/10 bg-card/35">
          <div className="mx-auto grid max-w-[1440px] gap-10 px-6 py-14 lg:grid-cols-[0.7fr_1.3fr] lg:px-10 lg:py-20">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">The pipeline</div>
              <h2 className="mt-4 max-w-sm text-3xl font-semibold tracking-[-0.04em]">Simple for everyone. Transparent for builders.</h2>
              <Link to="/lab" className="mt-6 inline-flex min-h-11 items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-ember hover:text-foreground">Open Gemma Lab <ArrowRight className="h-4 w-4" /></Link>
            </div>
            <ol className="grid gap-0 sm:grid-cols-2">
              {PIPELINE.map(({ icon: Icon, label, note }, index) => (
                <li key={label} className="flex gap-4 border-t border-white/10 py-5">
                  <span className="font-mono text-[11px] text-muted-foreground">{String(index + 1).padStart(2, "0")}</span>
                  <Icon className="mt-0.5 h-4 w-4 shrink-0 text-ember" aria-hidden="true" />
                  <div><div className="font-medium">{label}</div><div className="mt-1 text-sm text-muted-foreground">{note}</div></div>
                </li>
              ))}
            </ol>
          </div>
        </section>
      </main>
      <footer className="mx-auto flex max-w-[1440px] flex-col gap-3 px-6 py-8 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:flex-row sm:items-center sm:justify-between lg:px-10">
        <span>backend-powered pipeline</span>
        <span>media stays in configured run storage</span>
      </footer>
    </div>
  );
}
