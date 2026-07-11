import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { CircleDashed, AlertTriangle } from "lucide-react";

export function ProcessingState({ description = "GemmaClip is still preparing this stage. It will update automatically." }: { description?: ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-card p-8" role="status" aria-live="polite">
      <div className="h-1 overflow-hidden rounded-full bg-muted">
        <div className="h-full w-2/5 animate-pulse rounded-full bg-ember motion-reduce:animate-none" />
      </div>
      <div className="mt-5 font-display text-xl">Processing this stage</div>
      <p className="mt-2 max-w-lg text-sm text-muted-foreground">{description}</p>
    </div>
  );
}

export function EmptyState({
  title,
  description,
  action,
  className,
}: {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card/40 p-10 text-center",
        className,
      )}
    >
      <CircleDashed className="mb-3 h-8 w-8 text-muted-foreground/60" />
      <div className="font-display text-lg">{title}</div>
      {description && (
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  description,
  action,
  className,
}: {
  title?: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-xl border border-destructive/30 bg-destructive/5 p-10 text-center",
        className,
      )}
    >
      <AlertTriangle className="mb-3 h-8 w-8 text-destructive" />
      <div className="font-display text-lg">{title}</div>
      {description && (
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
