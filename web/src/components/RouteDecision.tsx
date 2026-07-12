import type { StructuredEvidence, ModelRoute } from "@/lib/types";
import { cn } from "@/lib/utils";
import { GitBranch } from "lucide-react";

interface Props {
  selected: Exclude<ModelRoute, "auto">;
  reason: string;
  auto?: boolean;
  audio: StructuredEvidence["audio"];
  className?: string;
  provider?: string;
  model?: string;
  modality?: "visual" | "audio_visual";
  audioFallbackOccurred?: boolean;
}

const ROUTE_LABEL: Record<Exclude<ModelRoute, "auto">, string> = {
  "gemma-4-26b-a4b": "Gemma 4 26B A4B",
  "gemma-4-12b-unified": "Gemma 4 12B Unified",
  "gemma-4-31b": "Gemma 4 31B",
};

export function RouteDecision({ selected, reason, auto, audio, className, provider, model, modality, audioFallbackOccurred }: Props) {
  const visualOnly = modality === "visual" || (modality === undefined && selected !== "gemma-4-12b-unified");
  return (
    <section className={cn("glass-panel overflow-hidden rounded-xl", className)} aria-label="Route decision">
      <div className="border-b border-white/10 px-4 py-4 sm:px-5">
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground"><GitBranch className="h-3.5 w-3.5 text-ember" /> Route decision</div>
      </div>
      <div className="grid gap-5 p-4 sm:p-5 md:grid-cols-[1fr_auto_1.2fr] md:items-center">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Provider</div>
          <div className="mt-2 break-words text-lg font-semibold uppercase tracking-tight">{provider ?? "configured"}</div>
        </div>
        <div className="hidden h-14 w-px bg-white/10 md:block" aria-hidden="true" />
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Model / modality</div>
          <div className="mt-2 text-xl font-semibold tracking-tight sm:text-2xl">{model ?? ROUTE_LABEL[selected]}</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {auto && <InstrumentLabel>automatic</InstrumentLabel>}
            <InstrumentLabel tone="ember">{visualOnly ? "visual only" : "visual + audio"}</InstrumentLabel>
            <InstrumentLabel>audio {audio.status}</InstrumentLabel>
            {audioFallbackOccurred && <InstrumentLabel tone="warning">audio dropped</InstrumentLabel>}
          </div>
          {modality && <div className="mt-3 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">actual modality: {modality.replace("_", " + ")}</div>}
        </div>
      </div>
      <div className="border-t border-white/10 px-4 py-4 sm:px-5">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Why this route</div>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted-foreground">{reason}</p>
      </div>
    </section>
  );
}

function InstrumentLabel({ children, tone = "default" }: { children: React.ReactNode; tone?: "default" | "ember" | "warning" }) {
  return <span className={cn("rounded-md border px-2 py-1 font-mono text-[10px] uppercase tracking-[0.14em]", tone === "ember" && "border-ember/40 bg-ember-soft text-ember", tone === "warning" && "border-warning/40 bg-warning/10 text-warning", tone === "default" && "border-white/10 bg-white/[0.03] text-muted-foreground")}>{children}</span>;
}
