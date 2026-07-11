import type { AudioSegment } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  segment: AudioSegment;
  durationSec: number;
  className?: string;
}

export function AudioWaveform({ segment, durationSec, className }: Props) {
  const { waveform } = segment;
  const startPct = (segment.startSec / durationSec) * 100;
  const endPct = (segment.endSec / durationSec) * 100;

  return (
    <div className={cn("rounded-xl border border-border bg-card p-4", className)}>
      <div className="mb-3 flex items-center justify-between font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
        <span>waveform · {durationSec.toFixed(1)}s</span>
        <span>rms {segment.rms.toFixed(3)}</span>
      </div>
      <div className="relative h-28 overflow-hidden rounded-md bg-background">
        <div
          className="absolute inset-y-0 z-0 bg-ember-soft/70"
          style={{ left: `${startPct}%`, width: `${endPct - startPct}%` }}
        />
        <div className="absolute inset-0 z-10 flex items-center gap-[2px] px-2">
          {waveform.map((v, i) => {
            const inside = i / waveform.length >= startPct / 100 && i / waveform.length <= endPct / 100;
            return (
              <span
                key={i}
                className={cn(
                  "flex-1 rounded-sm",
                  inside ? "bg-ember" : "bg-ink/25",
                )}
                style={{ height: `${Math.max(6, v * 100)}%` }}
              />
            );
          })}
        </div>
        <div
          className="pointer-events-none absolute inset-y-0 z-20 border-x border-ember"
          style={{ left: `${startPct}%`, width: `${endPct - startPct}%` }}
        />
      </div>
      <div className="mt-3 flex items-center justify-between font-mono text-xs text-muted-foreground">
        <span>selected {segment.startSec.toFixed(2)}s → {segment.endSec.toFixed(2)}s</span>
        <span>{(segment.endSec - segment.startSec).toFixed(2)}s window</span>
      </div>
    </div>
  );
}
