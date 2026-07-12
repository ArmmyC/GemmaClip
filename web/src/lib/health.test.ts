import { describe, expect, it } from "vitest";
import { getHealthActionState } from "./health";

describe("health action state", () => {
  it("disables Quick Caption and manual Lab actions when media tooling is unavailable", () => {
    const state = getHealthActionState({
      isError: false,
      isPending: false,
      data: { status: "unavailable", providersConfigured: true },
    });
    expect(state).toEqual({ serviceUnavailable: true, generationUnavailable: true });
  });

  it("keeps media inspection available while generation is degraded", () => {
    const state = getHealthActionState({
      isError: false,
      isPending: false,
      data: { status: "degraded", providersConfigured: false },
    });
    expect(state).toEqual({ serviceUnavailable: false, generationUnavailable: true });
  });
});
