import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { UploadDropzone } from "./UploadDropzone";

describe("UploadDropzone", () => {
  it("shows the selected file and ready state", () => {
    const onFile = vi.fn();
    const { container } = render(<UploadDropzone onFile={onFile} />);
    const file = new File(["video"], "scene.mp4", { type: "video/mp4" });
    fireEvent.change(container.querySelector('input[type="file"]')!, { target: { files: [file] } });
    expect(screen.getByText("scene.mp4")).toBeInTheDocument();
    expect(screen.getByText(/Ready \/ choose another/i)).toBeInTheDocument();
    expect(onFile).toHaveBeenCalledWith(file);
  });
});
