// Typed, promise-based mock API. Every function has the same shape a real
// FastAPI client would expose so swapping the implementation later is a
// one-file change (replace the bodies with `fetch(...)` calls).

import { makeDefaultRun } from "./mock-data";
import type {
  Run,
  FrameConfig,
  AudioConfig,
  EvidenceConfig,
  CaptionConfig,
  Experiment,
} from "./types";

const store = new Map<string, Run>();

// Seed a demo run so /lab/demo-run/* is always browsable in the prototype.
const DEMO_ID = "demo-run";
store.set(DEMO_ID, makeDefaultRun(DEMO_ID));

const delay = (ms = 240) => new Promise<void>((r) => setTimeout(r, ms));

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v));
}

export const api = {
  demoRunId: DEMO_ID,

  async createRun(filename = "uploaded_clip.mp4"): Promise<Run> {
    await delay();
    const id = `run-${Math.random().toString(36).slice(2, 8)}`;
    const run = makeDefaultRun(id);
    run.video.filename = filename;
    store.set(id, run);
    return clone(run);
  },

  async getRun(runId: string): Promise<Run> {
    await delay(80);
    const r = store.get(runId) ?? store.get(DEMO_ID)!;
    return clone(r);
  },

  async deleteRun(runId: string): Promise<void> {
    await delay();
    store.delete(runId);
  },

  async postMetadata(runId: string, preset: Run["preset"]): Promise<Run> {
    await delay();
    const r = store.get(runId) ?? makeDefaultRun(runId);
    r.preset = preset;
    r.stages.video = "complete";
    store.set(runId, r);
    return clone(r);
  },

  async postFrames(runId: string, config: FrameConfig): Promise<Run> {
    await delay(500);
    const r = store.get(runId) ?? makeDefaultRun(runId);
    r.frames.config = config;
    r.frames.frames = r.frames.frames
      .slice(0, config.totalFrames)
      .map((f) => ({ ...f, included: true }));
    r.stages.frames = "complete";
    r.stages.evidence = "invalidated";
    r.stages.captions = "invalidated";
    store.set(runId, r);
    return clone(r);
  },

  async postAudio(runId: string, config: AudioConfig): Promise<Run> {
    await delay(360);
    const r = store.get(runId) ?? makeDefaultRun(runId);
    r.audio.config = config;
    if (config.mode === "disabled") {
      r.audio.segment.energyCandidateFound = false;
      r.audio.segment.routeExplanation = "Audio analysis disabled by user.";
    }
    r.stages.audio = "complete";
    r.stages.evidence = "invalidated";
    r.stages.captions = "invalidated";
    store.set(runId, r);
    return clone(r);
  },

  async postEvidence(runId: string, config: EvidenceConfig): Promise<Run> {
    await delay(700);
    const r = store.get(runId) ?? makeDefaultRun(runId);
    r.evidence.config = config;
    r.evidence.result.selectedRoute =
      config.route === "auto"
        ? r.audio.segment.energyCandidateFound
          ? "gemma-4-12b-unified"
          : "gemma-4-26b-a4b"
        : config.route;
    r.evidence.result.routeReason =
      config.route === "auto"
        ? r.audio.segment.energyCandidateFound
          ? "A non-silent audio segment was available and sufficient runtime remained."
          : "No usable audio segment; routed to the visual-only model."
        : "Route selected manually.";
    r.stages.evidence = "complete";
    r.stages.captions = "invalidated";
    store.set(runId, r);
    return clone(r);
  },

  async postCaptions(runId: string, config: CaptionConfig): Promise<Run> {
    await delay(900);
    const r = store.get(runId) ?? makeDefaultRun(runId);
    r.captions.config = config;
    r.captions.results = r.captions.results.filter((c) =>
      config.styles.includes(c.style),
    );
    r.stages.captions = "complete";
    store.set(runId, r);
    return clone(r);
  },

  async postExperiment(runId: string, experiment: Experiment): Promise<Run> {
    await delay(400);
    const r = store.get(runId) ?? makeDefaultRun(runId);
    r.experiments = [...r.experiments.filter((e) => e.id !== experiment.id), experiment];
    store.set(runId, r);
    return clone(r);
  },
};

export type Api = typeof api;
