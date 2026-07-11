import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { startQuickUpload } from "@/lib/quick-flow";

afterEach(() => vi.restoreAllMocks());

describe("Quick Caption upload flow", () => {
  it("does not navigate, cleans up, and returns the original startup error", async () => {
    const run = { id: "run_aaaaaaaaaaaaaaaaaaaa" };
    const original = new ApiError("Gemma credentials are not configured.", 503);
    vi.spyOn(api, "createRun").mockResolvedValue(run as never);
    vi.spyOn(api, "startQuickCaption").mockRejectedValue(original);
    const remove = vi.spyOn(api, "deleteRun").mockResolvedValue();
    const navigate = vi.fn();
    await expect(startQuickUpload(new File(["x"], "clip.mp4"), navigate)).rejects.toBe(original);
    expect(remove).toHaveBeenCalledWith(run.id);
    expect(navigate).not.toHaveBeenCalled();
  });

  it("navigates only after startup succeeds and keeps the run", async () => {
    const run = { id: "run_aaaaaaaaaaaaaaaaaaaa" };
    vi.spyOn(api, "createRun").mockResolvedValue(run as never);
    vi.spyOn(api, "startQuickCaption").mockResolvedValue(run as never);
    const remove = vi.spyOn(api, "deleteRun");
    const navigate = vi.fn();
    await startQuickUpload(new File(["x"], "clip.mp4"), navigate);
    expect(remove).not.toHaveBeenCalled();
    expect(navigate).toHaveBeenCalledWith(run.id);
  });
});
