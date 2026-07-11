import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RouteDecision } from "./RouteDecision";

describe("RouteDecision", () => {
  it("shows actual Google visual fallback provenance", () => {
    render(
      <RouteDecision
        selected="gemma-4-31b"
        reason="Fireworks audio inference was unavailable, so frames-only fallback was used."
        provider="google"
        modality="visual"
        audioFallbackOccurred
        audio={{ status: "unavailable", speechPresent: false, language: null, transcript: null, visualConsistency: "unknown", captionSafeFacts: [] }}
      />,
    );
    expect(screen.getByText("Gemma 4 · 31B")).toBeInTheDocument();
    expect(screen.getByText("google")).toBeInTheDocument();
    expect(screen.getByText("audio dropped safely")).toBeInTheDocument();
    expect(screen.getByText("actual modality: visual")).toBeInTheDocument();
  });
});
