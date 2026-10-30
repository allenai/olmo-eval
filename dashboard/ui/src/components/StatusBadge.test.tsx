import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge, displayStatus } from "./primitives";

describe("displayStatus", () => {
  it("shows an interrupted run as interrupted, not as uploading", () => {
    expect(displayStatus({ status: "interrupted", stale: false, upload_state: "uploading" })).toBe("interrupted");
  });

  it("still shows a finished run whose upload is in progress as uploading", () => {
    expect(displayStatus({ status: "failed", stale: false, upload_state: "uploading" })).toBe("uploading");
  });
});

describe("StatusBadge", () => {
  it("labels an interrupted run and explains that it resumes", () => {
    render(<StatusBadge status="interrupted" />);
    const badge = screen.getByLabelText("Interrupted");
    expect(badge.textContent).toBe("Interrupted");
    expect(badge.getAttribute("title")).toContain("resumed job continues it");
  });
});
