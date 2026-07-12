import { Link, useRouterState } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import type { StageId, StageState } from "@/lib/types";
import { Check, AlertTriangle, CircleDashed, Loader2 } from "lucide-react";

const STAGES: { id: StageId; label: string }[] = [
  { id: "video", label: "Video" },
  { id: "frames", label: "Frames" },
  { id: "audio", label: "Audio" },
  { id: "evidence", label: "Evidence" },
  { id: "captions", label: "Captions" },
  { id: "compare", label: "Compare" },
];

interface Props {
  runId: string;
  states: Record<StageId, StageState>;
  status?: string;
  generationOutcome?: string | null;
}

export function PipelineStepper({ runId, states, status, generationOutcome }: Props) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <aside className="border-b border-white/10 bg-card/45 lg:sticky lg:top-16 lg:h-[calc(100dvh-4rem)] lg:border-b-0 lg:border-r lg:bg-card/25" aria-label="Lab pipeline">
      <div className="h-full px-4 py-3 sm:px-6 lg:px-5 lg:py-7">
        <div className="mb-4 flex items-center justify-between gap-4 lg:block">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Pipeline rail</div>
            <div className="mt-1.5 truncate font-mono text-xs text-foreground">RUN {shortRunId(runId)}</div>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground lg:hidden">
            <span className={cn("h-1.5 w-1.5 rounded-full", status === "ready" ? "bg-success" : status === "error" ? "bg-danger" : "bg-ember")} aria-hidden="true" />
            <span className="capitalize">{status ?? "processing"}</span>
          </div>
        </div>
        <nav aria-label="Run stages">
          <ol className="scrollbar-hide flex gap-1 overflow-x-auto pb-1 lg:block lg:space-y-1 lg:overflow-visible">
          {STAGES.map((s, i) => {
            const to = `/lab/$runId/${s.id}` as const;
            const active = pathname.endsWith(`/${s.id}`);
            const state = states[s.id];
            return (
              <li key={s.id} className="flex shrink-0 items-center gap-1 lg:block">
                <Link
                  to={to}
                  params={{ runId }}
                  aria-current={active ? "step" : undefined}
                  aria-label={`${s.label}, ${state}`}
                  className={cn(
                    "group flex min-h-11 items-center gap-2.5 rounded-md border border-transparent px-3 py-2 text-sm transition-colors",
                    active
                      ? "stage-active bg-accent text-foreground"
                      : "text-muted-foreground hover:border-white/10 hover:bg-white/[0.03] hover:text-foreground",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-6 w-6 shrink-0 items-center justify-center rounded-md border text-[10px] font-mono",
                      active
                        ? "border-ember/60 bg-ember-soft text-ember"
                        : "border-white/10 bg-background text-muted-foreground",
                    )}
                    role="img"
                    aria-label={`${s.label}: ${state}`}
                  >
                    {state === "complete" ? (
                      <Check className="h-3 w-3" />
                    ) : state === "invalidated" ? (
                      <AlertTriangle className="h-3 w-3 text-ember" />
                    ) : state === "error" ? (
                      "!"
                    ) : state === "active" ? (
                      <Loader2 className="h-3 w-3 animate-spin motion-reduce:animate-none" />
                    ) : (
                      <CircleDashed className="h-3 w-3" />
                    )}
                  </span>
                  <span className="font-mono text-[11px] uppercase tracking-[0.18em]">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span className="truncate">{s.label}</span>
                </Link>
                {i < STAGES.length - 1 && (
                  <span className="mx-1 h-px w-6 bg-border lg:ml-6 lg:block lg:h-3 lg:w-px" aria-hidden />
                )}
              </li>
            );
          })}
          </ol>
        </nav>
        <div className="mt-6 hidden border-t border-white/10 pt-4 lg:block">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Run status</div>
          <div className="mt-2 flex items-center gap-2 text-xs">
            <span className={cn("h-1.5 w-1.5 rounded-full", status === "ready" ? "bg-success" : status === "error" ? "bg-danger" : "bg-ember")} aria-hidden="true" />
            <span className="capitalize">{status ?? "processing"}</span>
          </div>
          {generationOutcome && <div className="mt-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">{generationOutcome.replaceAll("_", " ")}</div>}
        </div>
      </div>
    </aside>
  );
}

function shortRunId(runId: string) {
  return runId.length > 14 ? `${runId.slice(0, 7)}...${runId.slice(-5)}` : runId;
}
