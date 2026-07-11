import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Frame } from "@/lib/types";
import { FrameCard } from "./FrameCard";

const frame: Frame = {
  id: "frame-1",
  index: 1,
  timestampSec: 2.5,
  reason: "anchor",
  changeScore: 0.42,
  included: true,
  thumbnailUrl: "https://example.test/frame.jpg",
};

describe("FrameCard", () => {
  it("shows stored selection instead of a non-persistent include switch", () => {
    render(<FrameCard frame={frame} />);

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.getByText("selection stored by pipeline")).toBeInTheDocument();
  });
});
