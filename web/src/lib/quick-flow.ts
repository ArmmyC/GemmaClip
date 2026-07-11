import { createAndStartQuickRun } from "./api";
import type { Run } from "./types";

export async function startQuickUpload(file: File, onStarted: (runId: string) => void | Promise<void>): Promise<Run> {
  const run = await createAndStartQuickRun(file);
  await onStarted(run.id);
  return run;
}
