import { Link } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { Menu, X } from "lucide-react";
import { useState } from "react";

interface Props {
  variant?: "landing" | "app" | "lab";
  className?: string;
  labRunId?: string;
}

export function AppHeader({ variant = "app", className, labRunId }: Props) {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <header
      className={cn(
        "sticky top-0 z-40 w-full border-b border-white/10 bg-background/80 text-foreground shadow-[0_12px_40px_rgb(0_0_0_/_0.16)] backdrop-blur-xl",
        variant === "lab" && "text-paper",
        className,
      )}
    >
      <div className="mx-auto flex min-h-16 max-w-[1440px] items-center justify-between px-6 py-3 lg:px-10">
        <Link to="/" className="group flex items-center gap-3" onClick={() => setMenuOpen(false)}>
          <LogoMark />
          <div className="leading-none">
            <div className="font-display text-lg font-semibold tracking-[-0.035em]">GemmaClip</div>
            <div className={cn("mt-1 font-mono text-[9px] uppercase tracking-[0.19em] text-muted-foreground", variant === "lab" && "text-paper/55")}>
              pure Gemma pipeline
            </div>
          </div>
        </Link>
        <nav className="hidden items-center gap-1 md:flex" aria-label="Primary navigation">
          <NavLink to="/quick" dark={variant === "lab"}>Quick Caption</NavLink>
          {labRunId ? <NavLink to="/lab/$runId/video" params={{ runId: labRunId }} dark={variant === "lab"}>Gemma Lab</NavLink> : <NavLink to="/lab" dark={variant === "lab"}>Gemma Lab</NavLink>}
        </nav>
        <div className="flex items-center gap-2">
          <span className={cn("hidden items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:flex", variant === "lab" && "text-paper/55")}>
            <span className="h-1.5 w-1.5 rounded-full bg-success" aria-hidden="true" />
            pure Gemma pipeline
          </span>
          <button
            type="button"
            className="inline-flex h-11 w-11 items-center justify-center rounded-md border border-white/10 text-muted-foreground hover:bg-white/5 hover:text-foreground md:hidden"
            aria-label={menuOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>
      {menuOpen && (
        <nav className="border-t border-white/10 px-6 py-3 md:hidden" aria-label="Mobile navigation">
          <div className="grid gap-1">
            <NavLink to="/quick" dark={variant === "lab"} onClick={() => setMenuOpen(false)}>Quick Caption</NavLink>
            {labRunId ? <NavLink to="/lab/$runId/video" params={{ runId: labRunId }} dark={variant === "lab"} onClick={() => setMenuOpen(false)}>Gemma Lab</NavLink> : <NavLink to="/lab" dark={variant === "lab"} onClick={() => setMenuOpen(false)}>Gemma Lab</NavLink>}
          </div>
        </nav>
      )}
    </header>
  );
}

function NavLink({
  to,
  params,
  children,
  dark,
  onClick,
}: {
  to: string;
  params?: Record<string, string>;
  children: React.ReactNode;
  dark?: boolean;
  onClick?: () => void;
}) {
  return (
    <Link
      to={to}
      params={params}
      onClick={onClick}
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
