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
    <div className="mt-8 flex items-center justify-between gap-3 border-t border-border pt-6">
      <div>
        {prev && (
          <Button asChild variant="ghost" size="sm">
            <Link to={prev.to} params={{ runId }} className="gap-2">
              <ArrowLeft className="h-4 w-4" /> {prev.label}
            </Link>
          </Button>
        )}
      </div>
      <div>
        {next && (
          <Button asChild size="sm" className="gap-2">
            <Link to={next.to} params={{ runId }}>
              {next.label} <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        )}
      </div>
    </div>
  );
}
