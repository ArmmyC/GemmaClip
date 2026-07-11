import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { Run } from "./types";

export const runKey = (runId: string) => ["run", runId] as const;

export function useRun(runId: string) {
  return useQuery({
    queryKey: runKey(runId),
    queryFn: () => api.getRun(runId),
  });
}

export function useSetRun(runId: string) {
  const qc = useQueryClient();
  return (updater: (r: Run) => Run) => {
    qc.setQueryData<Run>(runKey(runId), (prev) => (prev ? updater(prev) : prev));
  };
}

export function useMutateRun<T>(runId: string, fn: (arg: T) => Promise<Run>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: (run) => qc.setQueryData(runKey(runId), run),
  });
}
