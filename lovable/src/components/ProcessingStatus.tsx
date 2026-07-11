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
    <div className={cn("rounded-xl border border-border bg-card p-6", className)}>
      <div className="mb-4 flex items-center justify-between">
        <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
          processing
        </div>
        {typeof pct === "number" && (
          <div className="font-mono text-xs text-muted-foreground">{pct}%</div>
        )}
      </div>
      {typeof pct === "number" && (
        <div className="mb-5 h-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-ember transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      <ol className="space-y-2">
        {steps.map((s, i) => {
          const done = i < activeIdx;
          const now = i === activeIdx;
          return (
            <li key={s} className="flex items-center gap-3">
              <span
                className={cn(
                  "flex h-6 w-6 items-center justify-center rounded-full border font-mono text-[10px]",
                  done && "border-ink bg-ink text-paper",
                  now && "border-ember bg-ember-soft text-ink",
                  !done && !now && "border-border text-muted-foreground",
                )}
              >
                {done ? "✓" : String(i + 1).padStart(2, "0")}
              </span>
              <span
                className={cn(
                  "text-sm",
                  now && "font-medium text-foreground",
                  !now && !done && "text-muted-foreground",
                )}
              >
                {s}
                {now && <span className="ml-2 inline-block animate-pulse text-ember">•••</span>}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
