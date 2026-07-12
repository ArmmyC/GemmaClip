import { createFileRoute, Outlet } from "@tanstack/react-router";
import { AppHeader } from "@/components/AppHeader";
import { PipelineStepper } from "@/components/PipelineStepper";
import { useRun } from "@/lib/hooks";
import { ErrorState } from "@/components/StateViews";
import { LabRunContent } from "@/components/LabRunContent";

export const Route = createFileRoute("/lab/$runId")({
  component: LabLayout,
});

function LabLayout() {
  const { runId } = Route.useParams();
  const { data: run, isLoading, error } = useRun(runId);

  return (
    <div className="min-h-[100dvh] bg-background text-foreground">
      <AppHeader />
      <div className="mx-auto grid min-h-[calc(100dvh-4rem)] max-w-[1440px] lg:grid-cols-[232px_minmax(0,1fr)]">
      {run && <PipelineStepper runId={runId} states={run.stages} status={run.status} generationOutcome={run.generationOutcome} />}
      <main className="min-w-0 px-4 py-7 sm:px-6 sm:py-9 lg:px-10 lg:py-12">
        {isLoading && (
          <div className="animate-pulse font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
            loading run…
          </div>
        )}
        {error ? (
          <ErrorState
            title="Could not load run"
            description={String((error as Error).message)}
          />
        ) : run?.mode === "quick" && run.status === "error" ? (
          <ErrorState
            title="Pipeline processing failed"
            description={run.error ?? "GemmaClip could not complete this run."}
          />
        ) : run ? (
          <LabRunContent run={run}>
            <Outlet />
          </LabRunContent>
        ) : null}
      </main>
      </div>
    </div>
  );
}
