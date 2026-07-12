import { useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { UploadCloud, Film } from "lucide-react";

interface Props {
  onFile?: (file: File) => void;
  compact?: boolean;
  className?: string;
  disabled?: boolean;
}

export function UploadDropzone({ onFile, compact, className, disabled = false }: Props) {
  const [hover, setHover] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(files: FileList | null) {
    if (!files?.[0]) return;
    setSelected(files[0]);
    onFile?.(files[0]);
  }

  return (
    <label
      onDragOver={(event) => { if (disabled) return; event.preventDefault(); setHover(true); }}
      onDragLeave={() => setHover(false)}
      onDrop={(event) => { if (disabled) return; event.preventDefault(); setHover(false); handleFiles(event.dataTransfer.files); }}
      className={cn(
        "group relative block overflow-hidden rounded-xl border border-white/10 bg-card/80 text-center transition-[border-color,background-color] duration-200",
        !disabled && "cursor-pointer",
        disabled && "cursor-not-allowed opacity-60",
        "hover:border-white/20 hover:bg-card",
        hover && "border-ember bg-ember-soft/20",
        compact ? "p-6" : "p-8 sm:p-12",
        className,
      )}
    >
      <input ref={inputRef} disabled={disabled} type="file" accept="video/mp4,video/webm,video/quicktime" className="sr-only" onChange={(event) => handleFiles(event.target.files)} />
      <div className="pointer-events-none absolute inset-0 signal-grid opacity-70" />
      <div className="relative flex flex-col items-center gap-4">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-white/10 bg-background text-ember">
          {selected ? <Film className="h-5 w-5" /> : <UploadCloud className="h-5 w-5 transition group-hover:-translate-y-0.5" />}
        </div>
        {selected ? (
          <div>
            <div className="font-mono text-sm text-foreground">{selected.name}</div>
            <div className="mt-2 flex items-center justify-center font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
              <span>{(selected.size / 1_000_000).toFixed(1)} MB</span>
              <span className="mx-2 text-muted-foreground/50">/</span>
              <span>{selected.type || "video file"}</span>
            </div>
            <div className="mt-3 inline-flex min-h-11 items-center rounded-md border border-ember/40 bg-ember-soft/20 px-3 font-mono text-[10px] uppercase tracking-[0.16em] text-ember">Ready / choose another</div>
          </div>
        ) : (
          <div>
            <div className="font-display text-2xl font-semibold tracking-tight">{disabled ? "Service unavailable" : "Drop video to begin"}</div>
            <div className="mt-2 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">MP4 / WEBM / MOV</div>
          </div>
        )}
      </div>
    </label>
  );
}
