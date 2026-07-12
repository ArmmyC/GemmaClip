import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({
  Link: ({ children, to, ...props }: { children: React.ReactNode; to: string }) => <a href={to} {...props}>{children}</a>,
}));

import { AppHeader } from "./AppHeader";

describe("AppHeader", () => {
  it("keeps the header focused on the brand and service status", () => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(["health"], { status: "ok", storage: "available", mediaTools: { ffmpeg: "available", ffprobe: "available" }, providersConfigured: true, jobManager: "available", version: "0.1.0" });
    render(<QueryClientProvider client={queryClient}><AppHeader /></QueryClientProvider>);
    expect(screen.getByRole("link", { name: /GemmaClip/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Gemma Lab" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Quick Caption" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /navigation/i })).not.toBeInTheDocument();
    expect(screen.getByText("Service ready")).toBeInTheDocument();
  });
});