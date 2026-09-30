import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { AnswerText, SourcesButton } from "./components";
import { runFixture } from "./execution/activityFixtures";

afterEach(cleanup);

it("renders answer headings while keeping citations interactive and content escaped", async () => {
  const onCitation = vi.fn();
  render(<AnswerText content={'## 收入增长 [1]\n\n<img src=x onerror=alert(1)>'} citationCount={1} onCitation={onCitation} />);
  expect(screen.getByRole("heading", { name: "收入增长 1" })).toBeTruthy();
  await userEvent.click(screen.getByRole("button", { name: "查看来源 1" }));
  expect(onCitation.mock.calls[0][0]).toBe(0);
  expect(document.querySelector("img")).toBeNull();
  expect(screen.getByText("<img src=x onerror=alert(1)>")).toBeTruthy();
});

it("keeps a source summary tied to the first referenced citation rather than the first stored citation", async () => {
  const onOpen = vi.fn();
  const citation = {
    ordinal: 1, index_chunk_id: null, document_id: "document-2", document_version_id: "version-2",
    document_display_name: "实际引用的报告", document_original_filename: "report.pdf", quoted_text: "原文",
    source_location: { surface_type: "page", surface_start: 12 }, score: null,
    modality: "text" as const, asset: null, matched_representations: [],
  };
  const run = runFixture({ citations: [{ ...citation, ordinal: 0, document_display_name: "未引用的报告" }, citation] });
  render(<SourcesButton content="结论。[2]" citations={run.citations} onOpen={onOpen} />);
  expect(screen.getByText("[2] 实际引用的报告")).toBeTruthy();
  expect(screen.queryByText("未引用的报告")).toBeNull();
  await userEvent.click(screen.getByRole("button", { name: "1 个来源" }));
  expect(onOpen.mock.calls[0][0]).toBe(1);
});
