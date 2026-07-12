import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";

interface Props {
  runId: string;
  prev?: { to: string; label: string };
  next?: { to: string; label: string };
}

export function PrevNext({ runId, prev, next }: Props) {
  return (
    <nav className="mt-10 flex flex-col gap-4 border-t border-white/10 pt-6 sm:flex-row sm:items-center sm:justify-between" aria-label="Lab stage navigation">
      <div className="min-w-0">
        {prev && (
          <Button asChild variant="ghost" size="sm" className="min-h-11 max-w-full justify-start gap-2 px-3 text-left">
            <Link to={prev.to} params={{ runId }} aria-label={`Previous stage: ${prev.label}`}>
              <ArrowLeft className="h-4 w-4" aria-hidden="true" /> <span className="truncate">{prev.label}</span>
            </Link>
          </Button>
        )}
      </div>
      <div className="flex items-center justify-between gap-3 sm:justify-end">
        <span className="hidden font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:inline">Continue through Lab</span>
        {next && (
          <Button asChild size="sm" className="min-h-11 gap-2">
            <Link to={next.to} params={{ runId }} aria-label={`Next stage: ${next.label}`}>
              {next.label} <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          </Button>
        )}
      </div>
    </nav>
  );
}
