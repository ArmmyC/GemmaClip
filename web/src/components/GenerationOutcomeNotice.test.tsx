import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GenerationOutcomeNotice } from "./GenerationOutcomeNotice";

describe("GenerationOutcomeNotice", () => {
  it("shows a truthful degraded warning for evidence fallback", () => {
    render(<GenerationOutcomeNotice outcome="evidence_fallback" />);
    expect(screen.getByText("Grounded fallback used")).toBeInTheDocument();
    expect(screen.getByText(/structured evidence/i)).toBeInTheDocument();
  });

  it("does not warn for model-generated output", () => {
    const { container } = render(<GenerationOutcomeNotice outcome="model_generated" />);
    expect(container).toBeEmptyDOMElement();
  });
});
