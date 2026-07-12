import { afterEach, describe, expect, it, vi } from "vitest";
import { copyText } from "./clipboard";

afterEach(() => vi.restoreAllMocks());

describe("copyText", () => {
  it("reports success only after the clipboard write resolves", async () => {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: vi.fn().mockResolvedValue(undefined) } });
    await expect(copyText("summary")).resolves.toBe(true);
  });

  it("reports failure when clipboard access is unavailable", async () => {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: undefined });
    await expect(copyText("summary")).resolves.toBe(false);
  });
});
