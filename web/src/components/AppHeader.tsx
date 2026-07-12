import { Link } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { ServiceHealthIndicator } from "@/components/ServiceHealth";

interface Props {
  variant?: "landing" | "app" | "lab";
  className?: string;
}

export function AppHeader({ variant = "app", className }: Props) {
  return (
    <header
      className={cn(
        "sticky top-0 z-40 w-full border-b border-white/10 bg-background/80 text-foreground shadow-[0_12px_40px_rgb(0_0_0_/_0.16)] backdrop-blur-xl",
        variant === "lab" && "text-paper",
        className,
      )}
    >
      <div className="mx-auto flex min-h-16 max-w-[1440px] items-center justify-between px-6 py-3 lg:px-10">
        <Link to="/" className="group flex items-center gap-3">
          <LogoMark />
          <div className="leading-none">
            <div className="font-display text-lg font-semibold tracking-[-0.035em]">GemmaClip</div>
            <div className={cn("mt-1 font-mono text-[9px] uppercase tracking-[0.19em] text-muted-foreground", variant === "lab" && "text-paper/55")}>
              pure Gemma pipeline
            </div>
          </div>
        </Link>
        <div className="flex items-center gap-2">
          <span className={cn("hidden items-center gap-2 text-muted-foreground sm:flex", variant === "lab" && "text-paper/55")}>
            <ServiceHealthIndicator />
          </span>
        </div>
      </div>
    </header>
  );
}

export function LogoMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "relative inline-flex h-9 w-9 items-end justify-center gap-0.5 overflow-hidden rounded-md border border-white/15 bg-foreground px-2 pb-2 pt-2 text-background",
        className,
      )}
    >
      <span className="h-2/5 w-1 rounded-sm bg-ember" />
      <span className="h-4/5 w-1 rounded-sm bg-ember" />
      <span className="h-3/5 w-1 rounded-sm bg-ember" />
      <span className="h-full w-1 rounded-sm bg-ember" />
    </span>
  );
}
