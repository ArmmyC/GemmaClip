import type { VideoMetadata } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  meta: VideoMetadata;
  className?: string;
}

function fmtDuration(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function fmtSize(bytes: number) {
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

export function VideoMetadataPanel({ meta, className }: Props) {
  const rows = [
    ["Filename", meta.filename],
    ["Duration", fmtDuration(meta.durationSec)],
    ["Resolution", `${meta.width} × ${meta.height}`],
    ["Frame rate", `${meta.fps} fps`],
    ["Codec", meta.codec],
    ["File size", fmtSize(meta.sizeBytes)],
    ["Audio stream", meta.hasAudioStream ? "present" : "not detected"],
  ] as const;
  return (
    <dl className={cn("divide-y divide-border rounded-xl border border-border bg-card", className)}>
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-center justify-between px-4 py-2.5">
          <dt className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            {k}
          </dt>
          <dd className="font-mono text-sm text-foreground">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
