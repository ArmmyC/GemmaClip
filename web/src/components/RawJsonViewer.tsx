import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Copy, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  data: unknown;
  className?: string;
  defaultOpen?: boolean;
}

export function RawJsonViewer({ data, className, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const text = JSON.stringify(data, null, 2);
  return (
    <div className={cn("glass-panel overflow-hidden rounded-xl", className)}>
      <button
        onClick={() => setOpen((o) => !o)}
        type="button"
        aria-expanded={open}
        aria-controls="structured-evidence-json"
        className="flex min-h-12 w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-white/[0.03]"
      >
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          raw · structured evidence
        </span>
        <ChevronDown
          className={cn("h-4 w-4 transition", open && "rotate-180")}
        />
      </button>
      {open && (
        <div id="structured-evidence-json" className="relative border-t border-white/10 bg-background">
          <Button
            variant="ghost"
            size="sm"
            className="absolute right-2 top-2 gap-1.5 font-mono text-[11px]"
            aria-label="Copy structured evidence JSON"
            onClick={() => navigator.clipboard?.writeText(text)}
          >
            <Copy className="h-3 w-3" /> copy
          </Button>
          <pre className="max-h-96 overflow-auto p-4 pr-20 font-mono text-xs leading-relaxed text-foreground">
            {text}
          </pre>
        </div>
      )}
    </div>
  );
}
