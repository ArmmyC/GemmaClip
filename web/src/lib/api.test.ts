import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, createAndStartQuickRun, fromBackendStyle, labPath, toBackendStyle, waitForRun } from "./api";

const response = (body: unknown, status = 200) => new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("real API client", () => {
  it("uploads the selected File with FormData", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ id: "run_aaaaaaaaaaaaaaaaaaaa" }, 201)); vi.stubGlobal("fetch", fetchMock);
    const file = new File(["video"], "clip.mp4", { type: "video/mp4" }); await api.createRun(file);
    const [, init] = fetchMock.mock.calls[0]; expect(init.method).toBe("POST"); expect(init.body).toBeInstanceOf(FormData); expect((init.body as FormData).get("video")).toBe(file);
  });

  it("polls processing status and resolves the completed run", async () => {
    const ready = { id: "run_aaaaaaaaaaaaaaaaaaaa", status: "ready" };
    const fetchMock = vi.fn().mockResolvedValueOnce(response({ id: ready.id, status: "processing", stages: {} })).mockResolvedValueOnce(response({ id: ready.id, status: "ready", stages: {} })).mockResolvedValueOnce(response(ready)); vi.stubGlobal("fetch", fetchMock);
    await expect(waitForRun(ready.id, { intervalMs: 0 })).resolves.toEqual(ready); expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("surfaces Quick Caption failures without demo fallback", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ id: "run_aaaaaaaaaaaaaaaaaaaa", status: "error", stages: {}, error: "Gemma failed safely." })));
    await expect(waitForRun("run_aaaaaaaaaaaaaaaaaaaa", { intervalMs: 0 })).rejects.toEqual(expect.objectContaining({ message: "Gemma failed safely." }));
  });

  it("maps a completed run to its real Lab URL", () => { expect(labPath("run_aaaaaaaaaaaaaaaaaaaa")).toBe("/lab/run_aaaaaaaaaaaaaaaaaaaa/video"); });

  it("maps caption style names explicitly", () => { expect(toBackendStyle("humorous-tech")).toBe("humorous_tech"); expect(fromBackendStyle("humorous_non_tech")).toBe("humorous-non-tech"); });

  it("does not hide production HTTP errors behind mock data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({ detail: "Run not found." }, 404)));
    await expect(api.getRun("run_aaaaaaaaaaaaaaaaaaaa")).rejects.toBeInstanceOf(ApiError);
  });

  it("deletes a newly uploaded run when startup fails and preserves the original error", async () => {
    const original = new ApiError("Missing credentials", 503);
    vi.spyOn(api, "createRun").mockResolvedValue({ id: "run_aaaaaaaaaaaaaaaaaaaa" } as never);
    vi.spyOn(api, "startQuickCaption").mockRejectedValue(original);
    const remove = vi.spyOn(api, "deleteRun").mockRejectedValue(new Error("cleanup failed"));
    await expect(createAndStartQuickRun(new File(["x"], "clip.mp4"))).rejects.toBe(original);
    expect(remove).toHaveBeenCalledWith("run_aaaaaaaaaaaaaaaaaaaa");
  });

  it("preserves the run after successful startup", async () => {
    const run = { id: "run_aaaaaaaaaaaaaaaaaaaa" };
    vi.spyOn(api, "createRun").mockResolvedValue(run as never);
    vi.spyOn(api, "startQuickCaption").mockResolvedValue(run as never);
    const remove = vi.spyOn(api, "deleteRun");
    await expect(createAndStartQuickRun(new File(["x"], "clip.mp4"))).resolves.toBe(run);
    expect(remove).not.toHaveBeenCalled();
  });

  it("does not delete when upload fails before a run exists", async () => {
    const original = new ApiError("Upload rejected", 413);
    vi.spyOn(api, "createRun").mockRejectedValue(original);
    const remove = vi.spyOn(api, "deleteRun");
    await expect(createAndStartQuickRun(new File(["x"], "clip.mp4"))).rejects.toBe(original);
    expect(remove).not.toHaveBeenCalled();
  });

  it("attempts cleanup for a startup conflict", async () => {
    const run = { id: "run_aaaaaaaaaaaaaaaaaaaa" };
    const original = new ApiError("Already processing", 409);
    vi.spyOn(api, "createRun").mockResolvedValue(run as never);
    vi.spyOn(api, "startQuickCaption").mockRejectedValue(original);
    const remove = vi.spyOn(api, "deleteRun").mockResolvedValue();
    await expect(createAndStartQuickRun(new File(["x"], "clip.mp4"))).rejects.toBe(original);
    expect(remove).toHaveBeenCalledWith(run.id);
  });
});
