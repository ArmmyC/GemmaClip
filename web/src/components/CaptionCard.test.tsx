import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CaptionCard } from "./CaptionCard";

describe("CaptionCard grounding context", () => {
  it("reports availability without claiming exact evidence use", () => {
    render(<CaptionCard caption={{
      id: "caption-formal", style: "formal", text: "A person walks through a room.",
      wordCount: 6, charCount: 30, status: "valid",
      groundingContext: { visualEvidenceAvailable: true, audioEvidenceAvailable: false },
    }} />);
    expect(screen.getByText("Grounding available")).toBeInTheDocument();
    expect(screen.getByText("visual evidence")).toBeInTheDocument();
    expect(screen.queryByText(/evidence used/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/caption-safe audio evidence/i)).not.toBeInTheDocument();
  });
});
