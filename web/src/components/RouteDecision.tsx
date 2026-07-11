import type { StructuredEvidence, ModelRoute } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { GitBranch } from "lucide-react";

interface Props {
  selected: Exclude<ModelRoute, "auto">;
  reason: string;
  auto?: boolean;
  audio: StructuredEvidence["audio"];
  className?: string;
  provider?: string;
  modality?: "visual" | "audio_visual";
  audioFallbackOccurred?: boolean;
}

const ROUTE_LABEL: Record<Exclude<ModelRoute, "auto">, string> = {
  "gemma-4-26b-a4b": "Gemma 4 · 26B A4B",
  "gemma-4-12b-unified": "Gemma 4 · 12B Unified",
  "gemma-4-31b": "Gemma 4 · 31B",
};

export function RouteDecision({ selected, reason, auto, audio, className, provider, modality, audioFallbackOccurred }: Props) {
  return (
    <div className={cn("relative overflow-hidden rounded-xl border border-border bg-card", className)}>
      <div className="pointer-events-none absolute inset-0 ember-glow opacity-40" />
      <div className="relative grid gap-6 p-5 md:grid-cols-[1fr_1px_1fr]">
        <div>
          <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            <GitBranch className="h-3.5 w-3.5" /> selected route
          </div>
          <div className="mt-2 font-display text-2xl leading-tight">
            {ROUTE_LABEL[selected]}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {auto && <Badge variant="outline">auto-routed</Badge>}
            <Badge className="border-ember/40 bg-ember-soft text-ink">
              {audio.status !== "unavailable" && selected === "gemma-4-12b-unified" ? "visual + audio" : "visual only"}
            </Badge>
            <Badge variant="secondary">audio {audio.status}</Badge>
            {provider && <Badge variant="outline">{provider}</Badge>}
            {audioFallbackOccurred && <Badge variant="outline">audio dropped safely</Badge>}
          </div>
        </div>
        <div className="hidden bg-border md:block" />
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            reason
          </div>
          <p className="mt-2 text-pretty text-sm leading-relaxed">{reason}</p>
          {modality && <p className="mt-3 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">actual modality: {modality.replace("_", " + ")}</p>}
        </div>
      </div>
    </div>
  );
}
