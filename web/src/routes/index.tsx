import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowRight, AudioLines, Braces, Clapperboard, Layers3 } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Button } from "@/components/ui/button";
import { startQuickUpload } from "@/lib/quick-flow";
import { createAndProbeManualRun } from "@/lib/api";
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
        <section aria-labelledby="landing-title" className="relative min-h-[calc(100dvh-4rem)] overflow-hidden signal-glow">
          <div className="pointer-events-none absolute inset-0 signal-grid opacity-80" />
          <div className="relative mx-auto grid min-h-[calc(100dvh-4rem)] max-w-[1440px] gap-12 px-6 py-12 sm:py-16 lg:grid-cols-[minmax(0,1.05fr)_minmax(360px,0.82fr)] lg:items-center lg:gap-16 lg:px-10 lg:py-10">
            <div className="max-w-3xl">
              <div className="mb-7 flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                <span className="h-px w-8 bg-ember" aria-hidden="true" />
                video captioning / Gemma 4
              </div>
              <h1 id="landing-title" className="max-w-3xl font-display text-4xl font-semibold leading-[1.02] tracking-[-0.03em] text-balance sm:text-6xl lg:text-7xl">
                <span className="block">Grounded captions.</span>
                <span className="block text-muted-foreground">Every decision visible.</span>
              </h1>
              <p className="mt-7 max-w-xl text-base leading-relaxed text-muted-foreground sm:text-lg">
                Drop a video. GemmaClip selects important moments, checks useful audio, builds structured evidence, and writes captions with Gemma.
              </p>
              <div className="mt-8 flex flex-wrap gap-x-5 gap-y-2 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground" aria-label="Quick Caption preset">
                <span className="inline-flex items-center gap-2"><span className="h-1.5 w-1.5 rounded-full bg-ember" aria-hidden="true" />Balanced preset</span>
                <span>6 hybrid frames</span>
                <span>audio auto</span>
              </div>
            </div>

            <div className="glass-panel rounded-2xl p-3 sm:p-4 lg:translate-y-2">
              <UploadDropzone className="min-h-[260px] sm:min-h-[320px] lg:min-h-[360px]" onFile={setFile} disabled={Boolean(serviceUnavailable)} />
              <div className="flex flex-col gap-3 px-3 pb-3 pt-4 sm:flex-row sm:px-4">
                <Button size="lg" className="min-h-11 flex-1 gap-2 rounded-md px-5" onClick={generate} disabled={!file || busy || generationUnavailable} title={generationUnavailable ? "Gemma generation is not configured or the service is unavailable." : undefined}>
                  {busy ? "Starting run" : "Generate captions"} <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Button>
                <Button size="lg" variant="outline" className="min-h-11 flex-1 gap-2 rounded-md border-input px-5" onClick={openLab} disabled={!file || busy || Boolean(serviceUnavailable)}>
                  Inspect in Gemma Lab
                </Button>
              </div>
              {error && <p role="alert" className="px-3 pb-4 text-sm text-danger sm:px-4">{error}</p>}
            </div>
          </div>
        </section>

        <section aria-labelledby="pipeline-title" className="border-y border-border bg-card/35">
          <div className="mx-auto grid max-w-[1440px] gap-10 px-6 py-14 lg:grid-cols-[0.7fr_1.3fr] lg:px-10 lg:py-20">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">What happens next</div>
              <h2 id="pipeline-title" className="mt-4 max-w-sm text-3xl font-semibold leading-tight tracking-[-0.035em]">Simple for everyone. Transparent for builders.</h2>
              <p className="mt-4 max-w-md text-sm leading-relaxed text-muted-foreground">The same run that gives you captions can open into Gemma Lab for a closer look at every observable stage.</p>
              <Link to="/lab" className="mt-6 inline-flex min-h-11 items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-ember hover:text-foreground">Explore Gemma Lab <ArrowRight className="h-4 w-4" aria-hidden="true" /></Link>
            </div>
            <ol className="grid gap-0 sm:grid-cols-2">
              {PIPELINE.map(({ icon: Icon, label, note }, index) => (
                <li key={label} className="flex gap-4 border-t border-border py-5">
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
