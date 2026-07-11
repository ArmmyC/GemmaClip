// Public browser-safe models returned by the GemmaClip FastAPI backend.

export type RunStatus =
  | "pending"
  | "processing"
  | "ready"
  | "error";

export type StageId =
  | "video"
  | "frames"
  | "audio"
  | "evidence"
  | "captions"
  | "compare";

export type StageState = "waiting" | "active" | "complete" | "invalidated" | "error";

export interface VideoMetadata {
  filename: string;
  durationSec: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  sizeBytes: number;
  hasAudioStream: boolean;
}

export type ProcessingPreset = "fast" | "balanced" | "maximum" | "custom";

export type FrameMethod = "uniform" | "aks-lite" | "hybrid";

export interface FrameConfig {
  method: FrameMethod;
  totalFrames: number;
  anchorCount: number;
  highChangeCount: number;
  minSpacingSec: number;
  changeSensitivity: number; // 0..1
}

export interface Frame {
  id: string;
  index: number;
  timestampSec: number;
  reason: "anchor" | "high-change" | "uniform";
  changeScore: number; // 0..1
  included: boolean;
  thumbnailUrl: string;
}

export interface ChangeSample {
  t: number;
  score: number;
}

export type AudioMode = "disabled" | "automatic" | "always";

export interface AudioConfig {
  mode: AudioMode;
  maxDurationSec: number;
  sampleRateHz: number;
  minRmsEnergy: number;
  strategy: "highest-energy" | "first-non-silent" | "custom-range";
  customStartSec?: number;
  customEndSec?: number;
}

export interface AudioSegment {
  startSec: number;
  endSec: number;
  rms: number;
  waveform: number[]; // normalized 0..1
  hasAudioStream: boolean;
  energyCandidateFound: boolean;
  routeExplanation: string;
  artifactAvailable: boolean;
  status: "usable" | "uncertain" | "silent" | "unavailable" | "failed";
}

export type ModelRoute =
  | "auto"
  | "gemma-4-26b-a4b"
  | "gemma-4-12b-unified";

export interface EvidenceConfig {
  route: ModelRoute;
  temperature: number;
  maxTokens: number;
  provider: string;
  showPromptStructure: boolean;
  showRawJson: boolean;
}

export interface StructuredEvidence {
  selectedRoute: Exclude<ModelRoute, "auto">;
  routeReason: string;
  scene: string;
  subjects: string[];
  actions: string[];
  setting: string;
  visibleObjects: string[];
  mood: string;
  cameraNotes: string;
  temporalProgression: string[];
  verifiedDescription: string;
  possibleMisreads: string[];
  unsupportedClaims: string[];
  styleHooks: string[];
  audio: {
    status: "usable" | "uncertain" | "silent" | "unavailable" | "failed";
    speechPresent: boolean;
    language: string | null;
    transcript: string | null;
    visualConsistency: "consistent" | "contradictory" | "unknown";
    captionSafeFacts: string[];
  };
}

export type CaptionStyle =
  | "formal"
  | "sarcastic"
  | "humorous-tech"
  | "humorous-non-tech"
  | "social"
  | "accessibility";

export interface CaptionConfig {
  model: "gemma-4-31b";
  temperature: number;
  minWords: number;
  maxWords: number;
  strictGrounding: boolean;
  audioEvidenceMode: "ignore" | "use-if-present" | "require";
  focusedRepair: boolean;
  styles: CaptionStyle[];
}

export interface Caption {
  id: string;
  style: CaptionStyle;
  text: string;
  wordCount: number;
  charCount: number;
  status: "valid" | "repaired";
  evidenceUsed: {
    visualScene: boolean;
    visibleAction: boolean;
    allowedAudioFact: boolean;
  };
}

export interface Experiment {
  id: string;
  label: string;
  frameMethod: FrameMethod;
  frameCount: number;
  timestamps: number[];
  audioMode: AudioMode;
  evidenceModel: Exclude<ModelRoute, "auto">;
  captionTemperature: number;
  runtimeMs: number;
  caption: string;
  style: CaptionStyle;
}

export interface Run {
  id: string;
  createdAt: string;
  status: RunStatus;
  video: VideoMetadata;
  preset: ProcessingPreset;
  frames: { config: FrameConfig; frames: Frame[]; changeSeries: ChangeSample[] };
  audio: { config: AudioConfig; segment: AudioSegment };
  evidence: { config: EvidenceConfig; result: StructuredEvidence };
  captions: { config: CaptionConfig; results: Caption[] };
  experiments: Experiment[];
  stages: Record<StageId, StageState>;
  activeStage?: StageId | null;
  progressMessage?: string | null;
  error?: string | null;
}

export interface RunStatusResponse {
  id: string;
  status: RunStatus;
  activeStage?: StageId | null;
  progressMessage?: string | null;
  stages: Record<StageId, StageState>;
  error?: string | null;
}

export interface ProgressEvent {
  stage: StageId;
  message: string;
  pct: number;
}
