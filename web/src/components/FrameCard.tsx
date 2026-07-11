import type { Frame } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Maximize2 } from "lucide-react";
import { useState } from "react";
import { Dialog, DialogContent, DialogTrigger, DialogTitle } from "@/components/ui/dialog";

interface Props {
  frame: Frame;
  onToggle?: (id: string, included: boolean) => void;
  className?: string;
}

function fmtTs(sec: number) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(2);
  return `${String(m).padStart(2, "0")}:${s.padStart(5, "0")}`;
}

export function FrameCard({ frame, onToggle, className }: Props) {
  const [included, setIncluded] = useState(frame.included);
  const reasonBadge =
    frame.reason === "anchor" ? (
      <Badge className="border-ember/40 bg-ember-soft text-ember">anchor</Badge>
    ) : frame.reason === "high-change" ? (
      <Badge className="border-lab/30 bg-lab-soft text-lab">high change</Badge>
    ) : (
      <Badge variant="secondary">uniform</Badge>
    );

  return (
    <div
      className={cn(
        "group overflow-hidden rounded-xl border border-white/10 bg-card transition-colors hover:border-white/20",
        included && "ring-1 ring-transparent hover:ring-ember/60",
        !included && "opacity-50 grayscale",
        className,
      )}
    >
      <div className="relative aspect-video overflow-hidden bg-muted">
        <img src={frame.thumbnailUrl} alt={`Frame ${frame.index} at ${fmtTs(frame.timestampSec)}, selected as ${frame.reason}`} className="h-full w-full object-cover" />
        <Dialog>
          <DialogTrigger asChild>
            <button
              className="absolute right-2 top-2 rounded-md bg-ink/60 p-1.5 text-paper opacity-0 backdrop-blur transition group-hover:opacity-100"
              aria-label="Preview full size"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </DialogTrigger>
          <DialogContent className="max-w-3xl">
            <DialogTitle className="font-mono text-xs uppercase tracking-[0.18em] text-muted-foreground">
              Frame {frame.index} · {fmtTs(frame.timestampSec)}
            </DialogTitle>
            <img src={frame.thumbnailUrl} alt={`Frame ${frame.index} preview at ${fmtTs(frame.timestampSec)}`} className="w-full rounded-md" />
          </DialogContent>
        </Dialog>
      </div>
      <div className="space-y-2 border-t border-white/10 p-4">
        <div className="flex items-center justify-between font-mono text-[11px] text-muted-foreground">
          <span>FRAME {String(frame.index).padStart(2, "0")}</span>
          <span>{fmtTs(frame.timestampSec)}</span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {reasonBadge}
          <Badge variant="outline" className="font-mono text-[10px]">
            Δ {frame.changeScore.toFixed(2)}
          </Badge>
        </div>
        <label className="flex items-center justify-between pt-1">
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            include
          </span>
          <Switch
            checked={included}
            onCheckedChange={(v) => {
              setIncluded(v);
              onToggle?.(frame.id, v);
            }}
          />
        </label>
      </div>
    </div>
  );
}
