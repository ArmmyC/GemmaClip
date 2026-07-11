import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { ArrowRight, ScanSearch } from "lucide-react";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Button } from "@/components/ui/button";
import { startQuickUpload } from "@/lib/quick-flow";

export const Route = createFileRoute("/lab")({ component: LabEntry });

function LabEntry() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function openLab() {
    if (!file) { setError("Choose a video first."); return; }
    setBusy(true); setError(null);
    try {
      await startQuickUpload(file, (runId) => navigate({ to: "/lab/$runId/video", params: { runId } }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The Lab run could not be created.");
    } finally { setBusy(false); }
  }

  return (
    <div className="min-h-[100dvh] bg-ink text-paper">
      <AppHeader variant="lab" />
      <main className="relative mx-auto grid max-w-7xl gap-12 overflow-hidden px-6 py-16 md:grid-cols-[0.9fr_1.1fr] md:items-center md:py-24">
        <div className="pointer-events-none absolute inset-0 grid-paper opacity-[0.08]" />
        <section className="relative">
          <ScanSearch className="h-9 w-9 text-ember" aria-hidden="true" />
          <h1 className="mt-7 max-w-xl font-display text-6xl leading-[0.95] tracking-tight md:text-8xl">Enter the glass-box pipeline.</h1>
          <p className="mt-6 max-w-lg text-lg leading-relaxed text-paper/70">Open a video as a live Gemma Lab run. Inspect its frames, audio decision, evidence route, and grounded captions.</p>
        </section>
        <section className="relative rounded-2xl border border-paper/15 bg-paper/[0.05] p-6 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-md md:p-8" aria-label="Create Gemma Lab run">
          <UploadDropzone onFile={setFile} />
          <Button className="mt-5 gap-2 rounded-full bg-paper px-6 text-ink hover:bg-paper/90" size="lg" onClick={openLab} disabled={busy}>
            {busy ? "Opening Lab..." : "Open in Gemma Lab"} <ArrowRight className="h-4 w-4" />
          </Button>
          {error && <p role="alert" className="mt-4 text-sm text-red-300">{error}</p>}
        </section>
      </main>
    </div>
  );
}
