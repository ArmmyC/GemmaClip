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
        "h-full rounded-lg border border-white/10 bg-card/80 p-4",
        tone === "warn" && "border-warning/35 bg-warning/5",
        tone === "info" && "border-lab/25 bg-lab-soft/5",
        className,
      )}
    >
      <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
        {label}
      </div>
      <div className="text-pretty text-sm leading-6">{children}</div>
    </div>
  );
}

export function EvidenceList({ items }: { items: string[] }) {
  if (!items.length) return <span className="text-muted-foreground">None recorded</span>;
  return (
    <ul className="space-y-1">
      {items.map((it, i) => (
        <li key={i} className="flex gap-2">
          <span className="mt-2 h-px w-3 shrink-0 bg-ember/70" />
          <span>{it}</span>
        </li>
      ))}
    </ul>
  );
}
