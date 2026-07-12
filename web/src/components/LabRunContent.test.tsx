import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { LabRunContent } from "./LabRunContent";
import { ProcessingState } from "./StateViews";

describe("LabRunContent", () => {
  afterEach(cleanup);
  it("shows the stored pipeline error instead of a nested processing state", () => {
    const run = {
      status: "error" as const,
      mode: "quick" as const,
      error: "Both evidence providers failed safely.",
      stages: { evidence: "error" as const },
    };

    render(
      <LabRunContent run={run}>
        <ProcessingState />
      </LabRunContent>,
    );

    expect(screen.getByText("Pipeline processing failed")).toBeInTheDocument();
    expect(screen.getByText(run.error)).toBeInTheDocument();
    expect(screen.queryByText("Processing this stage")).not.toBeInTheDocument();
  });

  it("keeps manual stage failures navigable for the stage retry UI", () => {
    const { queryByText } = render(
      <LabRunContent run={{ status: "error", error: "Frames failed", stageErrors: { frames: "Frames failed" } }}>
        <div>Frames retry content</div>
      </LabRunContent>,
    );
    expect(screen.getByText("Frames retry content")).toBeInTheDocument();
    expect(queryByText("Pipeline processing failed")).not.toBeInTheDocument();
  });
});
