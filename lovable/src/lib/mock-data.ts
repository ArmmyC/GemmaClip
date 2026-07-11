import type {
  Run,
  Frame,
  ChangeSample,
  Caption,
  Experiment,
  StructuredEvidence,
} from "./types";

// Deterministic pseudo-random for stable mock output
function rand(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

function makeThumb(index: number, hue: number, label: string) {
  // Inline SVG thumbnail — no external assets required.
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 180'>
  <defs>
    <linearGradient id='g${index}' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0' stop-color='hsl(${hue},55%,55%)'/>
      <stop offset='1' stop-color='hsl(${(hue + 40) % 360},60%,30%)'/>
    </linearGradient>
    <radialGradient id='v${index}' cx='0.5' cy='0.45' r='0.6'>
      <stop offset='0' stop-color='hsl(${(hue + 20) % 360},70%,70%)' stop-opacity='0.55'/>
      <stop offset='1' stop-color='transparent'/>
    </radialGradient>
  </defs>
  <rect width='320' height='180' fill='url(#g${index})'/>
  <circle cx='${140 + index * 8}' cy='${88 + (index % 3) * 6}' r='38' fill='url(#v${index})'/>
  <rect x='0' y='150' width='320' height='30' fill='rgba(0,0,0,0.35)'/>
  <text x='12' y='170' font-family='ui-monospace,monospace' font-size='12' fill='#f5efe4' letter-spacing='0.08em'>${label}</text>
</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const CHANGE_SERIES: ChangeSample[] = (() => {
  const r = rand(42);
  const out: ChangeSample[] = [];
  for (let i = 0; i <= 120; i++) {
    const t = (i / 120) * 24; // 24s clip
    const base = 0.15 + 0.25 * Math.sin(i / 6);
    const spike = i === 18 || i === 55 || i === 82 ? 0.6 : 0;
    out.push({ t, score: Math.min(1, Math.max(0.02, base + r() * 0.35 + spike)) });
  }
  return out;
})();

const FRAMES: Frame[] = [
  { index: 0, timestampSec: 0.4, reason: "anchor", changeScore: 0.22, hue: 22, label: "00:00.40" },
  { index: 1, timestampSec: 3.6, reason: "high-change", changeScore: 0.91, hue: 34, label: "00:03.60" },
  { index: 2, timestampSec: 7.9, reason: "uniform", changeScore: 0.44, hue: 190, label: "00:07.90" },
  { index: 3, timestampSec: 11.0, reason: "high-change", changeScore: 0.83, hue: 200, label: "00:11.00" },
  { index: 4, timestampSec: 16.4, reason: "anchor", changeScore: 0.31, hue: 260, label: "00:16.40" },
  { index: 5, timestampSec: 22.1, reason: "high-change", changeScore: 0.78, hue: 12, label: "00:22.10" },
].map((f, i) => ({
  id: `frame-${i}`,
  index: f.index,
  timestampSec: f.timestampSec,
  reason: f.reason as Frame["reason"],
  changeScore: f.changeScore,
  included: true,
  thumbnailUrl: makeThumb(i, f.hue, f.label),
}));

const WAVEFORM = (() => {
  const r = rand(9);
  const out: number[] = [];
  for (let i = 0; i < 220; i++) {
    const env = i < 40 ? i / 40 : i > 180 ? Math.max(0, (220 - i) / 40) : 1;
    const beat = 0.5 + 0.5 * Math.sin(i / 3);
    out.push(Math.min(1, env * (0.25 + r() * 0.55 * beat)));
  }
  return out;
})();

const EVIDENCE: StructuredEvidence = {
  selectedRoute: "gemma-4-12b-unified",
  routeReason: "A non-silent audio segment was available and sufficient runtime remained.",
  scene: "A person walks a golden retriever along a sunlit coastal boardwalk at golden hour.",
  subjects: ["adult person in beige jacket", "golden retriever on leash"],
  actions: ["walking", "occasional glance toward the water", "dog sniffing the boardwalk"],
  setting: "wooden boardwalk beside an ocean shoreline, low sun to camera-left",
  visibleObjects: ["leash", "wooden railing", "distant sailboats", "seagulls"],
  mood: "calm, unhurried, warm",
  cameraNotes: "handheld, slow forward dolly, shallow depth of field",
  temporalProgression: [
    "opens on wide shot of boardwalk",
    "cut to medium shot of subject and dog",
    "closes on backlit silhouette against water",
  ],
  verifiedDescription:
    "A person and a golden retriever walk together along a wooden coastal boardwalk during late-afternoon light.",
  possibleMisreads: ["shoreline might be a large lake rather than an ocean"],
  unsupportedClaims: ["no evidence of specific location or time of day beyond 'late afternoon'"],
  styleHooks: ["quiet companionship", "coastal light", "unhurried pace"],
  audio: {
    status: "analyzed",
    speechPresent: false,
    language: null,
    transcript: null,
    visualConsistency: "consistent",
    captionSafeFacts: [
      "ambient wind and distant waves are present",
      "no discernible speech",
    ],
  },
};

const CAPTIONS: Caption[] = [
  {
    id: "cap-1",
    style: "formal",
    text: "A person walks a golden retriever along a coastal boardwalk in late-afternoon light, with distant sailboats visible on calm water.",
    wordCount: 22,
    charCount: 138,
    status: "valid",
    evidenceUsed: { visualScene: true, visibleAction: true, allowedAudioFact: false },
  },
  {
    id: "cap-2",
    style: "humorous-tech",
    text: "Golden hour, golden retriever, zero packet loss — this dog is buffering good vibes at 60 fps along the boardwalk.",
    wordCount: 21,
    charCount: 118,
    status: "valid",
    evidenceUsed: { visualScene: true, visibleAction: true, allowedAudioFact: false },
  },
  {
    id: "cap-3",
    style: "social",
    text: "sunset walk with the best coworker 🐕✨ #goldenhour #coastalvibes",
    wordCount: 10,
    charCount: 68,
    status: "repaired",
    evidenceUsed: { visualScene: true, visibleAction: true, allowedAudioFact: false },
  },
  {
    id: "cap-4",
    style: "accessibility",
    text: "A person in a beige jacket walks slowly along a wooden boardwalk beside the ocean, holding a leash attached to a golden retriever. The dog pauses to sniff the boards. Warm, low sunlight comes from the left. Waves and wind are audible; no speech.",
    wordCount: 44,
    charCount: 244,
    status: "valid",
    evidenceUsed: { visualScene: true, visibleAction: true, allowedAudioFact: true },
  },
];

const EXPERIMENTS: Experiment[] = [
  {
    id: "exp-a",
    label: "Uniform · cold",
    frameMethod: "uniform",
    frameCount: 4,
    timestamps: [1.2, 7.8, 15.4, 22.0],
    audioMode: "disabled",
    evidenceModel: "gemma-4-26b-a4b",
    captionTemperature: 0.0,
    runtimeMs: 4820,
    style: "formal",
    caption:
      "A person walks a dog along a boardwalk beside water during the late afternoon.",
  },
  {
    id: "exp-b",
    label: "Hybrid · warm",
    frameMethod: "hybrid",
    frameCount: 6,
    timestamps: [0.4, 3.6, 7.9, 11.0, 16.4, 22.1],
    audioMode: "automatic",
    evidenceModel: "gemma-4-12b-unified",
    captionTemperature: 0.8,
    runtimeMs: 6710,
    style: "humorous-tech",
    caption:
      "Golden hour, golden retriever, zero packet loss — this dog is buffering good vibes at 60 fps along the boardwalk.",
  },
];

export function makeDefaultRun(id: string): Run {
  return {
    id,
    createdAt: new Date().toISOString(),
    status: "ready",
    video: {
      filename: "coastal_walk.mp4",
      durationSec: 24.0,
      width: 1920,
      height: 1080,
      fps: 30,
      codec: "H.264 / AAC",
      sizeBytes: 18_420_000,
      hasAudioStream: true,
    },
    preset: "balanced",
    frames: {
      config: {
        method: "hybrid",
        totalFrames: 6,
        anchorCount: 2,
        highChangeCount: 3,
        minSpacingSec: 1.5,
        changeSensitivity: 0.55,
      },
      frames: FRAMES,
      changeSeries: CHANGE_SERIES,
    },
    audio: {
      config: {
        mode: "automatic",
        maxDurationSec: 8,
        sampleRateHz: 16000,
        minRmsEnergy: 0.02,
        strategy: "highest-energy",
      },
      segment: {
        startSec: 6.2,
        endSec: 12.4,
        rms: 0.081,
        waveform: WAVEFORM,
        hasAudioStream: true,
        energyCandidateFound: true,
        routeExplanation:
          "Highest-energy non-silent window found between 6.2s and 12.4s (RMS 0.081).",
      },
    },
    evidence: {
      config: {
        route: "auto",
        temperature: 0.2,
        maxTokens: 1200,
        provider: "provider://gemma-gateway",
        showPromptStructure: false,
        showRawJson: false,
      },
      result: EVIDENCE,
    },
    captions: {
      config: {
        model: "gemma-4-31b",
        temperature: 0.7,
        minWords: 12,
        maxWords: 48,
        strictGrounding: true,
        audioEvidenceMode: "use-if-present",
        focusedRepair: true,
        styles: ["formal", "humorous-tech", "social", "accessibility"],
      },
      results: CAPTIONS,
    },
    experiments: EXPERIMENTS,
    stages: {
      video: "complete",
      frames: "complete",
      audio: "complete",
      evidence: "complete",
      captions: "complete",
      compare: "complete",
    },
  };
}

export const PROGRESS_MESSAGES = [
  "Inspecting video",
  "Selecting important moments",
  "Checking audio",
  "Building grounded evidence",
  "Writing captions",
];
