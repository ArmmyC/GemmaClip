import { Link, useRouterState } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import type { StageId, StageState } from "@/lib/types";
import { Check, AlertTriangle, CircleDashed } from "lucide-react";

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
}

export function PipelineStepper({ runId, states }: Props) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <div className="border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto max-w-7xl px-6">
        <ol className="scrollbar-hide flex gap-1 overflow-x-auto py-3">
          {STAGES.map((s, i) => {
            const to = `/lab/$runId/${s.id}` as const;
            const active = pathname.endsWith(`/${s.id}`);
            const state = states[s.id];
            return (
              <li key={s.id} className="flex shrink-0 items-center gap-1">
                <Link
                  to={to}
                  params={{ runId }}
                  className={cn(
                    "group flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition",
                    active
                      ? "bg-ink text-paper"
                      : "text-muted-foreground hover:bg-accent hover:text-foreground",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px] font-mono",
                      active
                        ? "border-paper/40 bg-paper/10 text-paper"
                        : "border-border bg-background text-muted-foreground",
                    )}
                  >
                    {state === "complete" ? (
                      <Check className="h-3 w-3" />
                    ) : state === "invalidated" ? (
                      <AlertTriangle className="h-3 w-3 text-ember" />
                    ) : state === "error" ? (
                      "!"
                    ) : (
                      <CircleDashed className="h-3 w-3" />
                    )}
                  </span>
                  <span className="font-mono text-[11px] uppercase tracking-[0.18em]">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span>{s.label}</span>
                </Link>
                {i < STAGES.length - 1 && (
                  <span className="mx-1 h-px w-6 bg-border" aria-hidden />
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
