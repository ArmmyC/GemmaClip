import { Link } from "@tanstack/react-router";
import { cn } from "@/lib/utils";

interface Props {
  variant?: "landing" | "app";
  className?: string;
}

export function AppHeader({ variant = "app", className }: Props) {
  return (
    <header
      className={cn(
        "w-full border-b border-border/70 bg-background/70 backdrop-blur-md",
        variant === "landing" && "border-transparent bg-transparent",
        className,
      )}
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
        <Link to="/" className="group flex items-center gap-3">
          <LogoMark />
          <div className="leading-none">
            <div className="font-display text-2xl tracking-tight">GemmaClip</div>
            <div className="mt-0.5 text-[10px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
              pure · gemma · pipeline
            </div>
          </div>
        </Link>
        <nav className="hidden items-center gap-1 md:flex">
          <NavLink to="/quick">Quick Caption</NavLink>
          <NavLink to="/quick">Gemma Lab</NavLink>
        </nav>
        <div className="flex items-center gap-2">
          <span className="hidden font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground sm:inline">
            v0.4 · prototype
          </span>
        </div>
      </div>
    </header>
  );
}

function NavLink({
  to,
  params,
  children,
}: {
  to: string;
  params?: Record<string, string>;
  children: React.ReactNode;
}) {
  return (
    <Link
      to={to}
      params={params}
      className="rounded-md px-3 py-1.5 text-sm text-muted-foreground transition hover:bg-accent hover:text-foreground"
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
