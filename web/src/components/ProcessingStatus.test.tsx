import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProcessingStatus } from "./ProcessingStatus";

describe("ProcessingStatus", () => {
  it("announces the active stage and truthful waiting states", () => {
    render(<ProcessingStatus steps={["Preparing video", "Selecting important moments", "Complete"]} active="Selecting important moments" />);
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
    expect(screen.getByText("Selecting important moments")).toBeInTheDocument();
    expect(screen.getByText("active")).toBeInTheDocument();
    expect(screen.getByText("waiting")).toBeInTheDocument();
    expect(screen.getByText("complete")).toBeInTheDocument();
  });
});
