import type { ProcessingPreset, Run, RunStatusResponse } from "./types";

const backendStyles = { formal: "formal", sarcastic: "sarcastic", "humorous-tech": "humorous_tech", "humorous-non-tech": "humorous_non_tech" } as const;
export const toBackendStyle = (style: keyof typeof backendStyles) => backendStyles[style];
export const fromBackendStyle = (style: string) => Object.entries(backendStyles).find(([, backend]) => backend === style)?.[0] ?? (() => { throw new Error(`Unsupported caption style: ${style}`); })();

const base = (import.meta.env.VITE_GEMMACLIP_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error { constructor(message: string, public status: number) { super(message); } }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, init);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { const body = await response.json() as { detail?: string }; if (body.detail) detail = body.detail; } catch { /* safe generic message */ }
    throw new ApiError(detail, response.status);
  }
  return response.status === 204 ? undefined as T : response.json() as Promise<T>;
}

export const mediaUrl = (runId: string) => `${base}/api/runs/${encodeURIComponent(runId)}/media/video`;
export const labPath = (runId: string) => `/lab/${encodeURIComponent(runId)}/video`;

export const api = {
  async createRun(file: File): Promise<Run> { const body = new FormData(); body.append("video", file); return request("/api/runs", { method: "POST", body }); },
  getRun: (id: string) => request<Run>(`/api/runs/${encodeURIComponent(id)}`),
  deleteRun: (id: string) => request<void>(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  postMetadata: (id: string, preset: ProcessingPreset) => request<Run>(`/api/runs/${encodeURIComponent(id)}/metadata`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ preset }) }),
  startQuickCaption: (id: string) => request<Run>(`/api/runs/${encodeURIComponent(id)}/quick-caption`, { method: "POST" }),
  getStatus: (id: string) => request<RunStatusResponse>(`/api/runs/${encodeURIComponent(id)}/status`),
  postFrames: (_id: string, _config: unknown): Promise<Run> => Promise.reject(new ApiError("Interactive reruns are coming in the next integration phase.", 501)),
  postAudio: (_id: string, _config: unknown): Promise<Run> => Promise.reject(new ApiError("Interactive reruns are coming in the next integration phase.", 501)),
  postEvidence: (_id: string, _config: unknown): Promise<Run> => Promise.reject(new ApiError("Interactive reruns are coming in the next integration phase.", 501)),
  postCaptions: (_id: string, _config: unknown): Promise<Run> => Promise.reject(new ApiError("Interactive reruns are coming in the next integration phase.", 501)),
  postExperiment: (_id: string, _config: unknown): Promise<Run> => Promise.reject(new ApiError("Experiment storage is coming in the next integration phase.", 501)),
};

export async function createAndStartQuickRun(file: File): Promise<Run> {
  const created = await api.createRun(file);
  try {
    await api.startQuickCaption(created.id);
  } catch (error) {
    await api.deleteRun(created.id).catch(() => undefined);
    throw error;
  }
  return created;
}

export async function waitForRun(id: string, options: { intervalMs?: number; signal?: AbortSignal; onStatus?: (s: RunStatusResponse) => void } = {}): Promise<Run> {
  const interval = options.intervalMs ?? 1000;
  for (;;) {
    if (options.signal?.aborted) throw new DOMException("Polling aborted", "AbortError");
    const status = await api.getStatus(id); options.onStatus?.(status);
    if (status.status === "ready") return api.getRun(id);
    if (status.status === "error") throw new ApiError(status.error ?? "Caption processing failed.", 422);
    await new Promise<void>((resolve, reject) => { const timer = setTimeout(resolve, interval); options.signal?.addEventListener("abort", () => { clearTimeout(timer); reject(new DOMException("Polling aborted", "AbortError")); }, { once: true }); });
  }
}
