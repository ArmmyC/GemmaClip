import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { StageId, StageState } from "@/lib/types";

vi.mock("@tanstack/react-router", () => ({
  Link: ({ children, to, params, className, ...props }: { children: React.ReactNode; to: string; params?: { runId?: string }; className?: string }) => (
    <a href={to.replace("$runId", params?.runId ?? "")} className={className} {...props}>
      {children}
    </a>
  ),
  useRouterState: ({ select }: { select: (state: { location: { pathname: string } }) => unknown }) =>
    select({ location: { pathname: "/lab/run-123/frames" } }),
}));

import { PipelineStepper } from "./PipelineStepper";

const states: Record<StageId, StageState> = {
  video: "complete",
  frames: "active",
  audio: "invalidated",
  evidence: "waiting",
  captions: "error",
  compare: "complete",
};

describe("PipelineStepper", () => {
  it("keeps all stage routes, active state, mobile overflow, and state semantics visible", () => {
    const { container } = render(
      <PipelineStepper runId="run-123" states={states} status="processing" generationOutcome={null} />,
    );

    for (const stage of ["Video", "Frames", "Audio", "Evidence", "Captions", "Compare"]) {
      expect(screen.getByRole("link", { name: new RegExp(stage) })).toHaveAttribute("href", `/lab/run-123/${stage.toLowerCase()}`);
    }

    expect(screen.getByRole("link", { name: /Frames/ })).toHaveClass("stage-active");
    expect(screen.getByRole("img", { name: "Video: complete" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Frames: active" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Audio: invalidated" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Evidence: waiting" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Captions: error" })).toBeInTheDocument();
    expect(container.querySelector("ol")).toHaveClass("overflow-x-auto");
  });
});
