import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/lib/hooks";
import type { HealthStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

function healthState(query: ReturnType<typeof useHealth>): HealthStatus | "checking" {
  if (query.isPending) return "checking";
  if (query.isError || query.data?.status === "unavailable") return "unavailable";
  return query.data?.status ?? "checking";
}

function healthCopy(state: HealthStatus | "checking") {
  if (state === "ok") return { label: "Service ready", detail: "The web API and media pipeline are available." };
  if (state === "degraded") return { label: "Limited configuration", detail: "Media inspection is available, but Gemma generation is not configured." };
  if (state === "unavailable") return { label: "Service unavailable", detail: "The web service could not be reached or its core storage is unavailable." };
  return { label: "Checking service", detail: "Checking storage, media tools, and safe service configuration." };
}

export function ServiceHealthIndicator() {
  const query = useHealth();
  const state = healthState(query);
  const copy = healthCopy(state);
  return (
    <span className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em]" title={copy.detail}>
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          state === "ok" && "bg-success",
          state === "degraded" && "bg-warning",
          state === "unavailable" && "bg-danger",
          state === "checking" && "bg-muted-foreground",
        )}
        aria-hidden="true"
      />
      <span>{copy.label}</span>
    </span>
  );
}

export function ServiceHealthNotice({ className }: { className?: string }) {
  const query = useHealth();
  const state = healthState(query);
  const copy = healthCopy(state);
  const retry = () => { void query.refetch(); };
  return (
    <section
      className={cn(
        "flex flex-wrap items-center justify-between gap-4 border-y px-4 py-3",
        state === "ok" && "border-success/20 bg-success/[0.04]",
        state === "degraded" && "border-warning/25 bg-warning/[0.05]",
        state === "unavailable" && "border-danger/30 bg-danger/[0.05]",
        state === "checking" && "border-white/10 bg-card/40",
        className,
      )}
      role={state === "unavailable" ? "alert" : "status"}
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <span
          className={cn(
            "mt-1.5 h-2 w-2 rounded-full",
            state === "ok" && "bg-success",
            state === "degraded" && "bg-warning",
            state === "unavailable" && "bg-danger",
            state === "checking" && "bg-muted-foreground",
          )}
          aria-hidden="true"
        />
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em]">{copy.label}</div>
          <p className="mt-1 text-sm text-muted-foreground">{copy.detail}</p>
        </div>
      </div>
      {state === "unavailable" && (
        <Button type="button" variant="outline" size="sm" className="min-h-11 gap-2" onClick={retry} disabled={query.isFetching}>
          <RefreshCw className={cn("h-3.5 w-3.5", query.isFetching && "animate-spin motion-reduce:animate-none")} />
          Retry health check
        </Button>
      )}
    </section>
  );
}
