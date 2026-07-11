import { createFileRoute, Link } from "@tanstack/react-router";
import { AppHeader } from "@/components/AppHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { Button } from "@/components/ui/button";
import { ArrowRight, Sparkles, Layers, Waves, Cpu, MessageSquareText, Beaker } from "lucide-react";
import { api } from "@/lib/api";
import { useNavigate } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: Landing,
});

function Landing() {
  const navigate = useNavigate();

  async function handleGenerate() {
    const run = await api.createRun();
    navigate({ to: "/quick", search: { runId: run.id } as never });
  }

  return (
    <div className="min-h-screen bg-background">
      <AppHeader variant="landing" />

      {/* HERO */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 grid-paper opacity-60" />
        <div className="pointer-events-none absolute -top-40 left-1/2 h-[520px] w-[900px] -translate-x-1/2 ember-glow" />
        <div className="relative mx-auto max-w-7xl px-6 pt-16 pb-20 md:pt-24 md:pb-28">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card/70 px-3 py-1 font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground backdrop-blur">
            <span className="h-1.5 w-1.5 rounded-full bg-ember" />
            pure gemma pipeline · v0.4 prototype
          </div>

          <h1 className="max-w-4xl font-display text-6xl leading-[0.95] tracking-tight text-balance md:text-8xl">
            Video captioning<br />
            powered by <em className="text-ember">pure Gemma.</em>
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-muted-foreground text-balance md:text-xl">
            Drop a video. Get grounded captions. Or open the lab and inspect every frame,
            audio segment, model route, and evidence object.
          </p>

          <div className="mt-10 grid gap-8 md:grid-cols-[1.15fr_1fr]">
            <div>
              <UploadDropzone />
              <div className="mt-5 flex flex-wrap items-center gap-3">
                <Button size="lg" className="gap-2 rounded-full px-6" onClick={handleGenerate}>
                  Generate captions <ArrowRight className="h-4 w-4" />
                </Button>
                <Button asChild size="lg" variant="ghost" className="gap-2 rounded-full">
                  <Link to="/lab/$runId/video" params={{ runId: api.demoRunId }}>
                    <Beaker className="h-4 w-4" /> Open Gemma Lab
                  </Link>
                </Button>
              </div>
              <p className="mt-4 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                mock backend · your file is not uploaded anywhere
              </p>
            </div>

            <aside className="relative overflow-hidden rounded-2xl border border-border bg-card p-6">
              <div className="pointer-events-none absolute inset-0 dot-paper opacity-40" />
              <div className="relative">
                <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                  the pipeline
                </div>
                <ol className="mt-4 space-y-3">
                  {[
                    { icon: Layers, label: "Frames", note: "anchor + high-change selection" },
                    { icon: Waves, label: "Audio", note: "energy-based segment window" },
                    { icon: Cpu, label: "Evidence", note: "Gemma 4 · route by content" },
                    { icon: MessageSquareText, label: "Captions", note: "grounded, styled, verifiable" },
                  ].map((s, i) => (
                    <li key={s.label} className="flex items-center gap-3 rounded-lg border border-border/60 bg-background/60 px-3 py-2.5">
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <s.icon className="h-4 w-4 text-ember" />
                      <div className="flex-1">
                        <div className="text-sm font-medium">{s.label}</div>
                        <div className="text-xs text-muted-foreground">{s.note}</div>
                      </div>
                    </li>
                  ))}
                </ol>
              </div>
            </aside>
          </div>
        </div>
      </section>

      {/* PROMISE STRIP */}
      <section className="border-y border-border bg-card/50">
        <div className="mx-auto grid max-w-7xl gap-8 px-6 py-14 md:grid-cols-2 md:items-center">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              product promise
            </div>
            <p className="mt-3 font-display text-4xl leading-tight md:text-5xl">
              Simple for everyone.<br />
              <span className="text-ember">Transparent for builders.</span>
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4 font-mono text-xs">
            <Stat k="stages" v="6" />
            <Stat k="models" v="Gemma 4" />
            <Stat k="secrets exposed" v="0" />
            <Stat k="hidden reasoning" v="never" />
          </div>
        </div>
      </section>

      {/* NERD SECTION */}
      <section className="relative overflow-hidden bg-ink text-paper">
        <div className="pointer-events-none absolute inset-0 grid-paper opacity-[0.06]" />
        <div className="relative mx-auto grid max-w-7xl gap-12 px-6 py-24 md:grid-cols-[1fr_1.1fr] md:items-center">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-paper/20 px-3 py-1 font-mono text-[11px] uppercase tracking-[0.22em] text-paper/60">
              <Sparkles className="h-3 w-3" /> for builders
            </div>
            <h2 className="mt-6 font-display text-6xl leading-[0.95] tracking-tight md:text-7xl">
              Are you a <em className="text-ember">nerd?</em>
            </h2>
            <p className="mt-5 max-w-lg text-paper/70 text-lg">
              Inspect every frame, audio segment, model route, evidence object,
              and generation setting. Change one variable, rerun, compare.
            </p>
            <div className="mt-8">
              <Button
                asChild
                size="lg"
                variant="secondary"
                className="gap-2 rounded-full bg-paper text-ink hover:bg-paper/90"
              >
                <Link to="/lab/$runId/video" params={{ runId: api.demoRunId }}>
                  Open Gemma Lab <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
          <div className="rounded-2xl border border-paper/15 bg-paper/[0.03] p-1">
            <PipelineDiagram />
          </div>
        </div>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-3 px-6 py-8 md:flex-row md:items-center">
          <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            gemmaclip · prototype · pure gemma
          </div>
          <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            no data leaves your browser
          </div>
        </div>
      </footer>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg border border-border bg-background px-4 py-3">
      <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{k}</div>
      <div className="mt-1 font-display text-2xl text-ink">{v}</div>
    </div>
  );
}

function PipelineDiagram() {
  const rows = [
    { t: "0.4s", label: "anchor", tone: "ember" },
    { t: "3.6s", label: "high change", tone: "lab" },
    { t: "7.9s", label: "uniform", tone: "muted" },
    { t: "11.0s", label: "high change", tone: "lab" },
    { t: "16.4s", label: "anchor", tone: "ember" },
    { t: "22.1s", label: "high change", tone: "lab" },
  ] as const;
  return (
    <div className="rounded-xl bg-ink p-6 font-mono text-xs text-paper/80">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-paper/50">// run.frames.hybrid</span>
        <span className="text-ember">6 frames selected</span>
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => (
          <div key={r.t} className="flex items-center gap-3">
            <span className="w-14 text-paper/50">{r.t}</span>
            <div className="h-1.5 flex-1 rounded-full bg-paper/10">
              <div
                className={
                  "h-full rounded-full " +
                  (r.tone === "ember"
                    ? "bg-ember w-1/4"
                    : r.tone === "lab"
                      ? "bg-[color:var(--lab)] w-3/4"
                      : "bg-paper/40 w-2/5")
                }
              />
            </div>
            <span className="w-24 text-right text-paper/60">{r.label}</span>
          </div>
        ))}
      </div>
      <div className="mt-6 grid grid-cols-3 gap-2 text-[10px] uppercase tracking-[0.2em] text-paper/50">
        <div>→ audio 6.2s–12.4s</div>
        <div>→ route 12B unified</div>
        <div>→ 4 captions</div>
      </div>
    </div>
  );
}
