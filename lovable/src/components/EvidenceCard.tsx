import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  label: string;
  children: ReactNode;
  className?: string;
  tone?: "default" | "warn" | "info";
}

export function EvidenceCard({ label, children, className, tone = "default" }: Props) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card p-4",
        tone === "warn" && "border-ember/40 bg-ember-soft/50",
        tone === "info" && "border-lab/30 bg-lab-soft/50",
        className,
      )}
    >
      <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        {label}
      </div>
      <div className="text-pretty text-sm leading-relaxed">{children}</div>
    </div>
  );
}

export function EvidenceList({ items }: { items: string[] }) {
  if (!items.length) return <span className="text-muted-foreground">—</span>;
  return (
    <ul className="space-y-1">
      {items.map((it, i) => (
        <li key={i} className="flex gap-2">
          <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-ink/50" />
          <span>{it}</span>
        </li>
      ))}
    </ul>
  );
}
