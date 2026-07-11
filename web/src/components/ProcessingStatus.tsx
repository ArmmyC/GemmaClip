import { cn } from "@/lib/utils";

interface Props {
  active?: string;
  steps: string[];
  className?: string;
  pct?: number;
}

export function ProcessingStatus({ active, steps, className, pct }: Props) {
  const activeIdx = active ? steps.indexOf(active) : -1;
  return (
    <div className={cn("glass-panel rounded-xl p-6", className)} role="status" aria-live="polite">
      <div className="mb-5 flex items-center justify-between">
        <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">processing run</div>
        {typeof pct === "number" && <div className="font-mono text-xs text-muted-foreground">{pct}%</div>}
      </div>
      {typeof pct === "number" && (
        <div className="mb-5 h-px overflow-hidden bg-white/10" aria-label={`Processing ${pct}%`}>
          <div className="h-full bg-ember transition-[width] duration-500 motion-reduce:transition-none" style={{ width: `${pct}%` }} />
        </div>
      )}
      <ol className="space-y-1">
        {steps.map((step, index) => {
          const done = index < activeIdx;
          const current = index === activeIdx;
          return (
            <li key={step} className="relative grid min-h-11 grid-cols-[28px_1fr_auto] items-center gap-3">
              {index < steps.length - 1 && <span className="absolute left-[13px] top-8 h-6 w-px bg-border" aria-hidden="true" />}
              <span className={cn("relative z-10 flex h-7 w-7 items-center justify-center rounded-md border font-mono text-[10px]", done && "border-success/50 bg-success/10 text-success", current && "border-ember bg-ember-soft text-ember", !done && !current && "border-border text-muted-foreground")}>
                {done ? "✓" : String(index + 1).padStart(2, "0")}
              </span>
              <span className={cn("text-sm", current && "font-medium text-foreground", !current && !done && "text-muted-foreground")}>{step}</span>
              <span className={cn("justify-self-end font-mono text-[10px] uppercase tracking-[0.14em]", done ? "text-success" : current ? "text-ember" : "text-muted-foreground/70")}>
                {done ? "complete" : current ? "active" : "waiting"}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
