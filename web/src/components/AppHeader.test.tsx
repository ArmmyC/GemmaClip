import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", () => ({
  Link: ({ children, to, onClick, params: _params, activeProps: _activeProps, ...props }: { children: React.ReactNode; to: string; onClick?: () => void; params?: unknown; activeProps?: unknown }) => <a href={to} onClick={onClick} {...props}>{children}</a>,
}));

import { AppHeader } from "./AppHeader";

describe("AppHeader", () => {
  it("keeps both product paths accessible, including mobile navigation", () => {
    render(<AppHeader />);
    expect(screen.getByRole("link", { name: "Quick Caption" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Gemma Lab" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
    expect(screen.getByRole("navigation", { name: "Mobile navigation" })).toBeInTheDocument();
  });
});
