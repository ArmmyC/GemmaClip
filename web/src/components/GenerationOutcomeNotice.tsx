import type { GenerationOutcome } from "@/lib/types";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

export function GenerationOutcomeNotice({ outcome, compact = false }: { outcome: GenerationOutcome | null; compact?: boolean }) {
  if (outcome === "model_generated") {
    return <div className={`${compact ? "mb-6" : ""} flex flex-wrap items-center gap-2.5 border-y border-success/20 py-3`} role="status"><CheckCircle2 className="h-4 w-4 text-success" aria-hidden="true" /><span className="font-mono text-[10px] uppercase tracking-[0.18em] text-success">Model-generated captions</span><span className="text-sm text-muted-foreground">Grounded output is ready.</span></div>;
  }
  if (outcome !== "evidence_fallback") return null;
  return (
    <div className={`${compact ? "mb-6" : ""} rounded-xl border border-warning/35 bg-warning/5 p-4`} role="status">
      <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-warning"><AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" /> Degraded generation outcome</div>
      <div className="mt-2 text-lg font-semibold">Grounded fallback used</div>
      <p className="mt-1 text-sm text-muted-foreground">Gemma produced structured evidence, but the final writing stage did not complete safely. These captions were generated from the grounded evidence.</p>
    </div>
  );
}
