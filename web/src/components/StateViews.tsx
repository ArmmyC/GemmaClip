import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { CircleDashed, AlertTriangle, Loader2 } from "lucide-react";
import type { StageId } from "@/lib/types";

export function ProcessingState({ description = "GemmaClip is still preparing this stage. It will update automatically." }: { description?: ReactNode }) {
  return (
    <div className="rounded-xl border border-white/10 bg-card/80 p-6 shadow-[0_8px_24px_rgb(0_0_0_/_0.12)] sm:p-8" role="status" aria-live="polite">
      <div className="flex items-center gap-3">
        <Loader2 className="h-4 w-4 animate-spin text-ember motion-reduce:animate-none" aria-hidden="true" />
        <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Stage in progress</div>
      </div>
      <div className="mt-4 font-display text-xl">Processing this stage</div>
      <p className="mt-2 max-w-lg text-sm text-muted-foreground">{description}</p>
      <div className="mt-6 h-px overflow-hidden bg-white/10">
        <div className="h-full w-2/5 animate-pulse rounded-full bg-ember motion-reduce:animate-none" />
      </div>
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
        "flex min-h-48 flex-col items-center justify-center rounded-xl border border-dashed border-white/15 bg-card/50 p-8 text-center",
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
        "flex min-h-48 flex-col items-center justify-center rounded-xl border border-danger/35 bg-danger/5 p-8 text-center",
        className,
      )}
      role="alert"
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

export function StageErrorState({
  stage,
  description,
  onRetry,
}: {
  stage: string;
  description?: ReactNode;
  onRetry?: () => void;
}) {
  return (
    <ErrorState
      title={`${stage} stage failed`}
      description={description ?? "This stage did not complete. Upstream artifacts remain available."}
      action={onRetry ? <button type="button" className="min-h-11 rounded-md border border-white/15 px-4 text-sm hover:bg-white/[0.04] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ember" onClick={onRetry}>Retry {stage}</button> : undefined}
    />
  );
}

export function StaleStageNotice({
  message = "This artifact is stale because an upstream setting changed. Run this stage again to refresh it.",
}: {
  message?: string;
}) {
  return <div className="flex items-start gap-2.5 rounded-lg border border-warning/30 bg-warning/5 px-3 py-2.5 text-sm text-warning" role="status"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" /><span>{message}</span></div>;
}

const DEPENDENTS: Record<Extract<StageId, "frames" | "audio" | "evidence" | "captions">, StageId[]> = {
  frames: ["evidence", "captions", "compare"],
  audio: ["evidence", "captions", "compare"],
  evidence: ["captions", "compare"],
  captions: ["compare"],
};

const LABELS: Record<StageId, string> = {
  video: "Video",
  frames: "Frames",
  audio: "Audio",
  evidence: "Evidence",
  captions: "Captions",
  compare: "Compare",
};

export function InvalidationPreview({ stage }: { stage: Extract<StageId, "frames" | "audio" | "evidence" | "captions"> }) {
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-warning/25 bg-warning/[0.04] px-3 py-2.5 text-sm text-warning" role="status">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <span>Running {LABELS[stage]} will refresh: {DEPENDENTS[stage].map((item) => LABELS[item]).join(", ")}.</span>
    </div>
  );
}
