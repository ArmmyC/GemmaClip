import type { Caption, CaptionStyle } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Copy, RefreshCw, CheckCircle2, Wrench } from "lucide-react";

const STYLE_LABEL: Record<CaptionStyle, string> = {
  formal: "Formal",
  sarcastic: "Sarcastic",
  "humorous-tech": "Humorous · Tech",
  "humorous-non-tech": "Humorous · Non-Tech",
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
    <div className="group flex flex-col overflow-hidden rounded-xl border border-border bg-card transition hover:border-ink/30">
      <div className="flex items-start justify-between border-b border-border/60 bg-background/40 px-5 py-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            style
          </div>
          <div className="mt-0.5 font-display text-lg leading-none">
            {STYLE_LABEL[caption.style]}
          </div>
        </div>
        <Badge
          variant="outline"
          className={cn(
            "gap-1 font-mono text-[10px]",
            caption.status === "valid"
              ? "border-emerald-600/30 text-emerald-800"
              : "border-ember/40 text-ember",
          )}
        >
          {caption.status === "valid" ? (
            <CheckCircle2 className="h-3 w-3" />
          ) : (
            <Wrench className="h-3 w-3" />
          )}
          {caption.status}
        </Badge>
      </div>
      <div className="flex-1 px-5 py-4">
        <p className="text-pretty text-[15px] leading-relaxed">{caption.text}</p>
      </div>
      <div className="space-y-3 border-t border-border/60 bg-background/40 px-5 py-3">
        <div className="flex flex-wrap gap-1.5">
          {caption.evidenceUsed.visualScene && (
            <EvBadge>visual scene</EvBadge>
          )}
          {caption.evidenceUsed.visibleAction && (
            <EvBadge>visible action</EvBadge>
          )}
          {caption.evidenceUsed.allowedAudioFact && (
            <EvBadge>allowed audio fact</EvBadge>
          )}
        </div>
        <div className="flex items-center justify-between">
          <span className="font-mono text-[11px] text-muted-foreground">
            {caption.wordCount}w · {caption.charCount}c
          </span>
          <div className="flex items-center gap-1">
            <Button
              size="sm"
              variant="ghost"
              className="gap-1.5"
              onClick={() => {
                navigator.clipboard?.writeText(caption.text);
                onCopy?.(caption.text);
              }}
            >
              <Copy className="h-3.5 w-3.5" /> copy
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="gap-1.5"
              onClick={() => onRegenerate?.(caption.id)}
            >
              <RefreshCw className="h-3.5 w-3.5" /> regenerate
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function EvBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-lab/30 bg-lab-soft px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-ink">
      {children}
    </span>
  );
}
