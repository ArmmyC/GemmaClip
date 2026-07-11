import { Link } from "@tanstack/react-router";
import { cn } from "@/lib/utils";

interface Props {
  variant?: "landing" | "app" | "lab";
  className?: string;
  labRunId?: string;
}

export function AppHeader({ variant = "app", className, labRunId }: Props) {
  return (
    <header
      className={cn(
        "w-full border-b border-border/70 bg-background/70 backdrop-blur-md",
        variant === "landing" && "border-transparent bg-transparent",
        variant === "lab" && "border-paper/10 bg-ink/80 text-paper",
        className,
      )}
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
        <Link to="/" className="group flex items-center gap-3">
          <LogoMark />
          <div className="leading-none">
            <div className="font-display text-2xl tracking-tight">GemmaClip</div>
            <div className={cn("mt-0.5 text-[10px] font-mono uppercase tracking-[0.22em] text-muted-foreground", variant === "lab" && "text-paper/55")}>
              pure Gemma pipeline
            </div>
          </div>
        </Link>
        <nav className="hidden items-center gap-1 md:flex">
          <NavLink to="/quick" dark={variant === "lab"}>Quick Caption</NavLink>
          {labRunId ? <NavLink to="/lab/$runId/video" params={{ runId: labRunId }} dark={variant === "lab"}>Gemma Lab</NavLink> : <NavLink to="/lab" dark={variant === "lab"}>Gemma Lab</NavLink>}
        </nav>
        <div className="flex items-center gap-2">
          <span className={cn("hidden font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground sm:inline", variant === "lab" && "text-paper/55")}>visual intelligence lab</span>
        </div>
      </div>
    </header>
  );
}

function NavLink({
  to,
  params,
  children,
  dark,
}: {
  to: string;
  params?: Record<string, string>;
  children: React.ReactNode;
  dark?: boolean;
}) {
  return (
    <Link
      to={to}
      params={params}
      className={cn("rounded-md px-3 py-1.5 text-sm text-muted-foreground transition hover:bg-accent hover:text-foreground", dark && "text-paper/70 hover:bg-paper/10 hover:text-paper")}
      activeProps={{ className: "text-foreground bg-accent" }}
    >
      {children}
    </Link>
  );
}

export function LogoMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "relative inline-flex h-9 w-9 items-center justify-center rounded-md border border-ink/80 bg-ink text-paper",
        className,
      )}
    >
      <span className="font-display text-xl leading-none">g</span>
      <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-ember" />
    </span>
  );
}
