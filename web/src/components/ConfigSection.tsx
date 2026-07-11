import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  tone?: "default" | "lab" | "ember";
}

export function ConfigSection({
  title,
  description,
  actions,
  children,
  className,
  tone = "default",
}: Props) {
  return (
    <section
      className={cn(
        "rounded-xl border border-border bg-card p-5",
        tone === "lab" && "bg-lab-soft/40",
        tone === "ember" && "bg-ember-soft/40",
        className,
      )}
    >
      <header className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h3 className="font-display text-xl leading-tight tracking-tight">{title}</h3>
          {description && (
            <p className="mt-1 max-w-prose text-sm text-muted-foreground">
              {description}
            </p>
          )}
        </div>
        {actions && <div className="shrink-0">{actions}</div>}
      </header>
      <div className="space-y-4">{children}</div>
    </section>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <label className="flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
        <span>{label}</span>
        {hint && <span className="normal-case tracking-normal">{hint}</span>}
      </label>
      {children}
    </div>
  );
}
