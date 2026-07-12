import type { ReactNode } from "react";
import type { Run } from "@/lib/types";
import { ErrorState } from "@/components/StateViews";

export function LabRunContent({
  run,
  children,
}: {
  run: Pick<Run, "status" | "error" | "stageErrors" | "mode"> | undefined;
  children: ReactNode;
}) {
  if (run?.mode === "quick" && run.status === "error") {
    return (
      <ErrorState
        title="Pipeline processing failed"
        description={run.error ?? "GemmaClip could not complete this run."}
      />
    );
  }
  return run ? children : null;
}
