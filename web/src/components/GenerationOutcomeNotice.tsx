import type { GenerationOutcome } from "@/lib/types";

export function GenerationOutcomeNotice({ outcome, compact = false }: { outcome: GenerationOutcome | null; compact?: boolean }) {
  if (outcome !== "evidence_fallback") return null;
  return (
    <div className={`${compact ? "mb-6" : ""} rounded-xl border border-ember/40 bg-ember-soft/40 p-4`} role="status">
      <div className="font-display text-xl">Grounded fallback used</div>
      <p className="mt-1 text-sm text-muted-foreground">
        Gemma produced structured evidence, but the final writing stage did not complete safely. These captions were generated from the grounded evidence.
      </p>
    </div>
  );
}
