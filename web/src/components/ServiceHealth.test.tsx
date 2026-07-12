import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ServiceHealthNotice } from "./ServiceHealth";

function renderWithHealth(status: "ok" | "degraded" | "unavailable", mediaUnavailable = false) {
  const queryClient = new QueryClient();
  queryClient.setQueryData(["health"], {
    status,
    storage: status === "unavailable" ? "unavailable" : "available",
    mediaTools: { ffmpeg: mediaUnavailable ? "unavailable" : "available", ffprobe: "available" },
    providersConfigured: status === "ok",
    jobManager: "available",
    version: "0.1.0",
  });
  return render(<QueryClientProvider client={queryClient}><ServiceHealthNotice /></QueryClientProvider>);
}

describe("ServiceHealthNotice", () => {
  it("explains limited configuration without exposing credential details", () => {
    renderWithHealth("degraded");
    expect(screen.getByText("Limited configuration")).toBeInTheDocument();
    expect(screen.getByText(/Media inspection is available/)).toBeInTheDocument();
    expect(screen.queryByText(/API_KEY|FIREWORKS|GOOGLE/i)).not.toBeInTheDocument();
  });

  it("offers retry when the service is unavailable", () => {
    renderWithHealth("unavailable", true);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry health check" })).toBeInTheDocument();
    expect(screen.getByText("Service unavailable")).toBeInTheDocument();
  });
});
