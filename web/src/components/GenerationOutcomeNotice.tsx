import type { GenerationOutcome } from "@/lib/types";

export function GenerationOutcomeNotice({ outcome, compact = false }: { outcome: GenerationOutcome | null; compact?: boolean }) {
  if (outcome === "model_generated") {
    return <div className={`${compact ? "mb-6" : ""} flex items-center gap-3 border-y border-success/20 py-3`} role="status"><span className="h-1.5 w-1.5 rounded-full bg-success" aria-hidden="true" /><span className="font-mono text-[10px] uppercase tracking-[0.18em] text-success">Model-generated captions</span><span className="text-sm text-muted-foreground">Grounded output is ready.</span></div>;
  }
  if (outcome !== "evidence_fallback") return null;
  return (
    <div className={`${compact ? "mb-6" : ""} rounded-xl border border-warning/35 bg-warning/5 p-4`} role="status">
      <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-warning">Degraded generation outcome</div>
      <div className="mt-2 text-lg font-semibold">Grounded fallback used</div>
      <p className="mt-1 text-sm text-muted-foreground">Gemma produced structured evidence, but the final writing stage did not complete safely. These captions were generated from the grounded evidence.</p>
    </div>
  );
}
