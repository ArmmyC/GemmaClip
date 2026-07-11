import { createFileRoute, Outlet } from "@tanstack/react-router";
import { AppHeader } from "@/components/AppHeader";
import { PipelineStepper } from "@/components/PipelineStepper";
import { useRun } from "@/lib/hooks";
import { ErrorState } from "@/components/StateViews";

export const Route = createFileRoute("/lab/$runId")({
  component: LabLayout,
});

function LabLayout() {
  const { runId } = Route.useParams();
  const { data: run, isLoading, error } = useRun(runId);

  return (
    <div className="min-h-screen bg-background">
      <AppHeader />
      {run && <PipelineStepper runId={runId} states={run.stages} />}
      <main className="mx-auto max-w-7xl px-6 py-8">
        {isLoading && (
          <div className="animate-pulse font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
            loading run…
          </div>
        )}
        {error && (
          <ErrorState
            title="Could not load run"
            description={String((error as Error).message)}
          />
        )}
        {run && <Outlet />}
      </main>
    </div>
  );
}
