import type { Frame, ChangeSample } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  series: ChangeSample[];
  frames: Frame[];
  durationSec: number;
  className?: string;
}

export function FrameTimeline({ series, frames, durationSec, className }: Props) {
  const max = Math.max(...series.map((s) => s.score));
  return (
    <div className={cn("rounded-xl border border-border bg-card p-4", className)}>
      <div className="mb-3 flex items-center justify-between">
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
          <g className="text-ink">
            <path
              d={
                "M0,100 " +
                series
                  .map((s, i) => `L${(i / (series.length - 1)) * 1000},${100 - (s.score / max) * 92}`)
                  .join(" ") +
                " L1000,100 Z"
              }
              fill="url(#area)"
            />
            <path
              d={series
                .map((s, i) => `${i === 0 ? "M" : "L"}${(i / (series.length - 1)) * 1000},${100 - (s.score / max) * 92}`)
                .join(" ")}
              fill="none"
              stroke="currentColor"
              strokeOpacity="0.6"
              strokeWidth="1"
            />
          </g>
          {frames.map((f) => {
            const x = (f.timestampSec / durationSec) * 1000;
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
