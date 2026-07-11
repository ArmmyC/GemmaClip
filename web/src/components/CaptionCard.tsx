import type { Caption, CaptionStyle } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Copy, RefreshCw, CheckCircle2, Wrench } from "lucide-react";

const STYLE_LABEL: Record<CaptionStyle, string> = {
  formal: "Formal",
  sarcastic: "Sarcastic",
  "humorous-tech": "Humorous / Tech",
  "humorous-non-tech": "Humorous / Non-Tech",
  social: "Social Media",
  accessibility: "Accessibility Description",
};

interface Props {
  caption: Caption;
  onRegenerate?: (id: string) => void;
  onCopy?: (text: string) => void;
}

export function CaptionCard({ caption, onRegenerate, onCopy }: Props) {
  return (
    <article className="group flex flex-col overflow-hidden rounded-xl border border-white/10 bg-card transition-colors hover:border-white/20">
      <header className="flex items-start justify-between gap-4 border-b border-white/10 px-5 py-4">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">caption style</div>
          <div className="mt-1 font-display text-xl font-semibold tracking-tight">{STYLE_LABEL[caption.style]}</div>
        </div>
        <div className={cn("inline-flex items-center gap-1 rounded-md border px-2 py-1 font-mono text-[10px] uppercase tracking-[0.14em]", caption.status === "valid" ? "border-success/40 text-success" : "border-ember/40 text-ember")}>
          {caption.status === "valid" ? <CheckCircle2 className="h-3 w-3" /> : <Wrench className="h-3 w-3" />}
          {caption.status}
        </div>
      </header>
      <div className="flex-1 px-5 py-6"><p className="text-pretty text-[17px] leading-[1.65] text-foreground">{caption.text}</p></div>
      <footer className="space-y-4 border-t border-white/10 px-5 py-4">
        <div>
          <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Grounding available</div>
          <div className="flex flex-wrap gap-2">
            {caption.groundingContext.visualEvidenceAvailable && <EvidenceBadge>visual evidence</EvidenceBadge>}
            {caption.groundingContext.audioEvidenceAvailable && <EvidenceBadge>caption-safe audio</EvidenceBadge>}
            {!caption.groundingContext.visualEvidenceAvailable && !caption.groundingContext.audioEvidenceAvailable && <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">No evidence metadata</span>}
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="font-mono text-[11px] text-muted-foreground">{caption.wordCount} words / {caption.charCount} chars</span>
          <div className="flex items-center gap-1">
            <Button size="sm" variant="ghost" className="min-h-11 gap-1.5" onClick={() => { navigator.clipboard?.writeText(caption.text); onCopy?.(caption.text); }}><Copy className="h-3.5 w-3.5" /> Copy</Button>
            {onRegenerate && <Button size="sm" variant="ghost" className="min-h-11 gap-1.5" onClick={() => onRegenerate(caption.id)}><RefreshCw className="h-3.5 w-3.5" /> Regenerate</Button>}
          </div>
        </div>
      </footer>
    </article>
  );
}

function EvidenceBadge({ children }: { children: React.ReactNode }) {
  return <span className="inline-flex items-center rounded-md border border-lab/30 bg-lab-soft px-2 py-1 font-mono text-[10px] uppercase tracking-[0.14em] text-lab">{children}</span>;
}
