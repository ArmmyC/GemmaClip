import type { Frame, ChangeSample } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  series: ChangeSample[];
  frames: Frame[];
  durationSec: number;
  className?: string;
}

export function FrameTimeline({ series, frames, durationSec, className }: Props) {
  if (!series.length) {
    return (
      <div className={cn("glass-panel rounded-xl border-dashed p-6", className)} role="status">
        <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Visual change</div>
        <p className="mt-2 text-sm text-muted-foreground">The change series will appear after frame extraction completes.</p>
      </div>
    );
  }

  const max = Math.max(...series.map((s) => s.score), 0.001);
  const pointX = (index: number) => (index / Math.max(series.length - 1, 1)) * 1000;
  const timestampX = (timestampSec: number) => durationSec > 0 ? (timestampSec / durationSec) * 1000 : 0;
  return (
    <div className={cn("glass-panel rounded-xl p-4", className)} role="img" aria-label={`Visual change timeline for ${durationSec.toFixed(1)} seconds`}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          visual change · {durationSec.toFixed(1)}s
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-ember" /> anchor
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-lab" /> high change
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-muted-foreground/60" /> uniform
          </span>
        </div>
      </div>
      <div className="relative h-24">
        <svg viewBox="0 0 1000 100" preserveAspectRatio="none" className="h-full w-full">
          <defs>
            <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor="currentColor" stopOpacity="0.28" />
              <stop offset="1" stopColor="currentColor" stopOpacity="0" />
            </linearGradient>
          </defs>
          <g className="text-foreground">
            <path
              d={
                "M0,100 " +
                series
                  .map((s, i) => `L${pointX(i)},${100 - (s.score / max) * 92}`)
                  .join(" ") +
                " L1000,100 Z"
              }
              fill="url(#area)"
            />
            <path
              d={series
                .map((s, i) => `${i === 0 ? "M" : "L"}${pointX(i)},${100 - (s.score / max) * 92}`)
                .join(" ")}
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.6"
              strokeWidth="1"
            />
          </g>
          {frames.map((f) => {
            const x = timestampX(f.timestampSec);
            const color =
              f.reason === "anchor"
                ? "var(--ember)"
                : f.reason === "high-change"
                  ? "var(--lab)"
                  : "var(--muted-foreground)";
            return (
              <g key={f.id}>
                <line x1={x} x2={x} y1={0} y2={100} stroke={color} strokeOpacity="0.35" strokeDasharray="2 3" />
                <circle cx={x} cy={100 - f.changeScore * 92} r={4} fill={color} />
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>0.0s</span>
        <span>{(durationSec / 2).toFixed(1)}s</span>
        <span>{durationSec.toFixed(1)}s</span>
      </div>
    </div>
  );
}
