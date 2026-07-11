import { useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { UploadCloud, Film } from "lucide-react";

interface Props {
  onFile?: (file: File) => void;
  compact?: boolean;
  className?: string;
}

export function UploadDropzone({ onFile, compact, className }: Props) {
  const [hover, setHover] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(files: FileList | null) {
    if (!files || !files[0]) return;
    setSelected(files[0]);
    onFile?.(files[0]);
  }

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={(e) => {
        e.preventDefault();
        setHover(false);
        handleFiles(e.dataTransfer.files);
      }}
      className={cn(
        "group relative block cursor-pointer overflow-hidden rounded-xl border-2 border-dashed border-border bg-card/60 p-8 text-center transition",
        "hover:border-ink/50 hover:bg-card",
        hover && "border-ember bg-ember-soft/40",
        compact ? "py-8" : "py-16",
        className,
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/webm,video/quicktime"
        className="sr-only"
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="pointer-events-none absolute inset-0 dot-paper opacity-40" />
      <div className="relative flex flex-col items-center gap-4">
        <div className="flex h-14 w-14 items-center justify-center rounded-full border border-border bg-background">
          {selected ? (
            <Film className="h-6 w-6 text-ember" />
          ) : (
            <UploadCloud className="h-6 w-6 text-ink/70 transition group-hover:text-ember" />
          )}
        </div>
        {selected ? (
          <div>
            <div className="font-mono text-sm">{selected.name}</div>
            <div className="mt-1 text-xs text-muted-foreground">
              {(selected.size / 1_000_000).toFixed(1)} MB · ready to process
            </div>
          </div>
        ) : (
          <div>
            <div className="font-display text-2xl">Upload or drop your video</div>
            <div className="mt-1 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
              MP4 · WEBM · MOV
            </div>
          </div>
        )}
      </div>
    </label>
  );
}
