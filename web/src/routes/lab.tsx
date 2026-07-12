import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowRight, ScanSearch } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Button } from "@/components/ui/button";
import { createAndProbeManualRun } from "@/lib/api";
import { startQuickUpload } from "@/lib/quick-flow";
import { ServiceHealthNotice } from "@/components/ServiceHealth";
import { useHealth } from "@/lib/hooks";

export const Route = createFileRoute("/lab")({ component: LabEntry });

function LabEntry() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const health = useHealth();
  const serviceUnavailable = health.isError || health.data?.status === "unavailable";
  const generationUnavailable = serviceUnavailable || health.isPending || health.data?.providersConfigured === false;

  async function openLab() {
    if (!file) { setError("Choose a video first."); return; }
    setBusy(true); setError(null);
    try {
      const run = await createAndProbeManualRun(file, "balanced");
      navigate({ to: "/lab/$runId/video", params: { runId: run.id } });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The Lab run could not be created.");
    } finally { setBusy(false); }
  }

  async function runAutomatically() {
    if (!file) { setError("Choose a video first."); return; }
    setBusy(true); setError(null);
    try {
      await startQuickUpload(file, (runId) => navigate({ to: "/quick", search: { runId } as never }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The automatic run could not be created.");
    } finally { setBusy(false); }
  }

  return (
    <div className="min-h-[100dvh] bg-background text-foreground signal-glow">
      <AppHeader variant="lab" />
      <main className="relative mx-auto grid max-w-7xl gap-12 overflow-hidden px-6 py-16 md:grid-cols-[0.9fr_1.1fr] md:items-center md:py-24">
        <div className="pointer-events-none absolute inset-0 grid-paper opacity-[0.08]" />
        <section className="relative">
          <ScanSearch className="h-9 w-9 text-ember" aria-hidden="true" />
          <h1 className="mt-7 max-w-xl font-display text-5xl font-semibold leading-[0.96] tracking-[-0.055em] md:text-7xl">Open the glass-box pipeline.</h1>
          <p className="mt-6 max-w-lg text-lg leading-relaxed text-muted-foreground">Inspect frame selection, audio routing, structured evidence, and every generated caption from one stored run.</p>
          <div className="mt-8 flex flex-wrap gap-x-5 gap-y-2 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground"><span>Frames</span><span>Audio</span><span>Evidence</span><span>Captions</span></div>
        </section>
        <section className="glass-panel relative rounded-2xl p-2 md:p-3" aria-label="Create Gemma Lab run">
          <ServiceHealthNotice className="mb-2 rounded-xl" />
          <UploadDropzone onFile={setFile} />
          <div className="mt-4 flex flex-wrap gap-3">
            <Button className="min-h-11 gap-2 rounded-lg px-6" size="lg" onClick={openLab} disabled={busy || !file || Boolean(serviceUnavailable)}>
              {busy ? "Opening Lab" : "Open manual Lab"} <ArrowRight className="h-4 w-4" />
            </Button>
            <Button variant="outline" className="min-h-11 gap-2 rounded-lg px-6" size="lg" onClick={runAutomatically} disabled={busy || !file || generationUnavailable}>
              Run automatically
            </Button>
          </div>
          {error && <p role="alert" className="mt-4 text-sm text-danger">{error}</p>}
        </section>
      </main>
    </div>
  );
}
