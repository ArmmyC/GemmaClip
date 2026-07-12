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
        "rounded-xl border border-white/10 bg-card/80 p-4 shadow-[0_8px_24px_rgb(0_0_0_/_0.12)] sm:p-5",
        tone === "lab" && "border-lab/25 bg-lab-soft/[0.07]",
        tone === "ember" && "border-ember/25 bg-ember-soft/[0.07]",
        className,
      )}
    >
      <header className="mb-5 flex flex-col gap-4 border-b border-white/10 pb-5 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
        <div>
          <h3 className="font-display text-lg font-semibold leading-tight tracking-[-0.02em]">{title}</h3>
          {description && (
            <p className="mt-1 max-w-prose text-sm text-muted-foreground">
              {description}
            </p>
          )}
        </div>
        {actions && <div className="w-full shrink-0 sm:w-auto">{actions}</div>}
      </header>
      <div className="space-y-5">{children}</div>
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
