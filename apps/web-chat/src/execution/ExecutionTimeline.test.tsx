import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ApiClient } from "../api/client";
import { ToolCallRow } from "./ToolCallRow";
import { ExecutionTimeline } from "./ExecutionTimeline";
import { ActivitySourcePreview } from "./ActivitySourcePreview";
import { applyActivity, disconnectActivity, emptyActivity } from "./activityState";
import { eventFixture, RUN_ID, runFixture, snapshotFixture, sourceFixture, stepFixture } from "./activityFixtures";
const client = {} as ApiClient;
const onCitation = vi.fn();
afterEach(() => { cleanup(); vi.useRealTimers(); });
describe("execution timeline", () => {
  it("shows parallel calls, safe text, independent failures and expandable inputs", async () => {
    const unsafe = '<img src=x onerror="alert(1)">';
    const first = stepFixture({ queries: [unsafe] });
    const second = stepFixture({ step_id: "step_2", ordinal: 2, seq: 2, name: "keyword_search", status: "failed", ended_offset_ms: 500, result_code: "keyword_unavailable" });
    let activity = applyActivity(emptyActivity(RUN_ID), RUN_ID, 1, eventFixture(first));
    activity = applyActivity(activity, RUN_ID, 1, eventFixture(second));
    const { container } = render(<ExecutionTimeline run={runFixture()} activity={activity} progress={null} client={client} onCitation={onCitation} />);
    expect(screen.getByText("2 个工具并行")).toBeTruthy();
    expect(screen.getByText("关键词索引当前不可用")).toBeTruthy();
    expect(container.querySelector("img")).toBeNull();
    const toggle = screen.getByRole("button", { name: /语义检索.*semantic_search/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    await userEvent.click(toggle);
    expect(screen.getByText("查询")).toBeTruthy();
    expect(document.getElementById(toggle.getAttribute("aria-controls")!)?.textContent).toContain(unsafe);
  });
  it("reconciles live partial previews with saved details and keeps expansion", () => {
    const activity = applyActivity(emptyActivity(RUN_ID), RUN_ID, 1, eventFixture(stepFixture({ queries: ["预览…"], details_truncated: true }), 2));
    const { rerender } = render(<ExecutionTimeline run={runFixture()} activity={activity} progress={null} client={client} onCitation={onCitation} />);
    const saved = snapshotFixture([stepFixture({ seq: 8, status: "succeeded", ended_offset_ms: 1600, sources: [sourceFixture], returned_count: 3, new_evidence_count: 2 })]);
    rerender(<ExecutionTimeline run={runFixture({ status: "completed", activities: [saved] })} activity={activity} progress={null} client={client} onCitation={onCitation} />);
    expect(screen.queryByText("预览…")).toBeNull();
    expect(screen.getByText("返回 3 条资料 · 合并新增 2 条")).toBeTruthy();
    expect(screen.getByRole("button", { name: /回答过程/ }).getAttribute("aria-expanded")).toBe("true");
  });
  it("freezes disconnected steps and offers follow after scrolling away", () => {
    vi.useFakeTimers();
    const state = applyActivity(emptyActivity(RUN_ID), RUN_ID, 1, eventFixture(), Date.now());
    const { container } = render(<ExecutionTimeline run={runFixture()} activity={disconnectActivity(state, RUN_ID)} progress={null} client={client} onCitation={onCitation} />);
    expect(screen.getByText("未确认完成")).toBeTruthy();
    expect(screen.queryByText("正在执行")).toBeNull();
    const before = screen.getByRole("button", { name: /回答过程/ }).textContent;
    act(() => { vi.advanceTimersByTime(5000); });
    expect(screen.getByRole("button", { name: /回答过程/ }).textContent).toBe(before);
    const scroll = container.querySelector(".activity-scroll")!;
    Object.defineProperties(scroll, { scrollHeight: { value: 900 }, clientHeight: { value: 300 } });
    fireEvent.scroll(scroll, { target: { scrollTop: 50 } });
    fireEvent.click(screen.getByRole("button", { name: "回到最新进度" }));
    expect(scroll.scrollTop).toBe(900);
  });
  it("separates attempts with unique details and does not invent missing history", async () => {
    const saved = [snapshotFixture([stepFixture({ status: "failed", ended_offset_ms: 300 })]), snapshotFixture([stepFixture({ status: "succeeded", ended_offset_ms: 400 })], 2)];
    const { container, rerender } = render(<ExecutionTimeline run={runFixture({ status: "completed", attempt: 2, activities: saved })} activity={null} progress={null} client={client} onCitation={onCitation} />);
    await userEvent.click(screen.getByRole("button", { name: /回答过程/ }));
    await userEvent.click(screen.getByText("第 1 次尝试 · 已结束"));
    const buttons = screen.getAllByRole("button", { name: /语义检索.*semantic_search/ });
    await userEvent.click(buttons[0]); await userEvent.click(buttons[1]);
    const ids = [...container.querySelectorAll("[id]")].map(node => node.id);
    expect(new Set(ids).size).toBe(ids.length);
    rerender(<ExecutionTimeline run={runFixture({ status: "completed" })} activity={null} progress={null} client={client} onCitation={onCitation} />);
    expect(screen.getByText("这次回答没有保存逐步执行记录。")).toBeTruthy();
    expect(screen.queryByText("核验证据并回答")).toBeNull();
  });
});
describe("retrieved source preview", () => {
  function preview(api: Partial<ApiClient>) {
    const trigger = document.createElement("button"); document.body.append(trigger);
    const onClose = vi.fn();
    const view = render(<ActivitySourcePreview client={api as ApiClient} run={runFixture()} source={sourceFixture} trigger={trigger} onClose={onClose} />);
    return { ...view, trigger, onClose };
  }
  it("never substitutes a newer document version", async () => {
    const getDocumentChunks = vi.fn();
    const view = preview({ getDocument: vi.fn().mockResolvedValue({ kb_id: "kb-1", current_version: { id: "new-version" }, index: { index_revision_id: "index-1" } }), getDocumentChunks });
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(getDocumentChunks).not.toHaveBeenCalled();
    view.unmount(); expect(document.activeElement).toBe(view.trigger); view.trigger.remove();
  });
  it("loads the exact chunk, keeps text inert, and supports Escape", async () => {
    const getDocumentChunks = vi.fn().mockResolvedValue({ document_id: sourceFixture.document_id, document_version_id: sourceFixture.document_version_id, index_revision_id: "index-1", items: [{ id: sourceFixture.index_chunk_id, excluded_at: null, content: "原始收入片段 <script>private()</script>" }], next_cursor: null });
    const view = preview({ getDocument: vi.fn().mockResolvedValue({ kb_id: "kb-1", current_version: { id: sourceFixture.document_version_id }, index: { index_revision_id: "index-1" } }), getDocumentChunks });
    await waitFor(() => expect(within(screen.getByRole("dialog")).getByText(/原始收入片段/)).toBeTruthy());
    expect(view.container.querySelector("script")).toBeNull();
    fireEvent.keyDown(window, { key: "Escape" }); expect(view.onClose).toHaveBeenCalledOnce();
    view.unmount(); view.trigger.remove();
  });
});


describe("multi-library source identity", () => {
  it("opens a source using its own frozen library revision", async () => {
    const trigger = document.createElement("button"); document.body.append(trigger);
    const source = {...sourceFixture, knowledge_base_id: "kb-b", knowledge_base_name: "库 B", index_revision_id: "revision-b"};
    const run = runFixture({knowledge_base_id:null,index_revision_id:null,knowledge_bases:[{knowledge_base_id:"kb-a",name:"库 A",index_revision_id:"revision-a",status:"ready"},{knowledge_base_id:"kb-b",name:"库 B",index_revision_id:"revision-b",status:"ready"}]});
    const api = {getDocument:vi.fn().mockResolvedValue({kb_id:"kb-b",current_version:{id:source.document_version_id},index:{index_revision_id:"revision-b"}}),getDocumentChunks:vi.fn().mockResolvedValue({document_id:source.document_id,document_version_id:source.document_version_id,index_revision_id:"revision-b",items:[{id:source.index_chunk_id,content:"来自库 B 的同名文档",excluded_at:null}],next_cursor:null})} as unknown as ApiClient;
    const view=render(<ActivitySourcePreview client={api} run={run} source={source} trigger={trigger} onClose={vi.fn()} />);
    expect(await screen.findByText("来自库 B 的同名文档")).toBeTruthy();
    expect(screen.getByText("库 B")).toBeTruthy();
    view.unmount();trigger.remove();
  });
  it.each(["wrong-library","wrong-revision"])("rejects %s before reading chunk contents",async mismatch => {
    const trigger=document.createElement("button");document.body.append(trigger);
    const source={...sourceFixture,knowledge_base_id:mismatch==="wrong-library"?"kb-a":"kb-b",index_revision_id:mismatch==="wrong-revision"?"revision-a":"revision-b"};
    const run=runFixture({knowledge_base_id:null,index_revision_id:null,knowledge_bases:[{knowledge_base_id:"kb-b",name:"库 B",index_revision_id:"revision-b",status:"ready"}]});
    const getDocumentChunks=vi.fn();
    const api={getDocument:vi.fn().mockResolvedValue({kb_id:"kb-b",current_version:{id:source.document_version_id},index:{index_revision_id:"revision-b"}}),getDocumentChunks} as unknown as ApiClient;
    const view=render(<ActivitySourcePreview client={api} run={run} source={source} trigger={trigger} onClose={vi.fn()} />);
    expect(await screen.findByRole("alert")).toBeTruthy();expect(getDocumentChunks).not.toHaveBeenCalled();view.unmount();trigger.remove();
  });
});


describe("timeline citation links", () => {
  it.each(["matching", "wrong-library", "wrong-revision"])("keeps displayed numbering and source identity for %s", async variant => {
    const source = {...sourceFixture, knowledge_base_id: "kb-b", index_revision_id: "revision-b"};
    const citation = {ordinal: 2, index_chunk_id: source.index_chunk_id, document_id: source.document_id,
      document_version_id: source.document_version_id, document_display_name: source.title,
      document_original_filename: "deployment.txt", quoted_text: "库 B 的原文", source_location: {}, score: null,
      modality: "text" as const, asset: null, matched_representations: ["text"],
      knowledge_base_id: variant === "wrong-library" ? "kb-a" : "kb-b",
      index_revision_id: variant === "wrong-revision" ? "revision-a" : "revision-b"};
    const step = stepFixture({status: "succeeded", sources: [source]});
    const onOpen = vi.fn();
    render(<ToolCallRow step={step} steps={[step]} live={false} elapsedMs={500} run={runFixture({citations:[citation]})}
      attempt={1} onCitation={onOpen} onSource={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", {name: /语义检索/}));
    if (variant === "matching") {
      await userEvent.click(screen.getByRole("button", {name: "引用来源 3"}));
      expect(onOpen).toHaveBeenCalledWith(2, expect.any(HTMLButtonElement));
    } else {
      expect(screen.queryByRole("button", {name: /引用来源/})).toBeNull();
      expect(screen.getByRole("button", {name: "查看片段"})).toBeTruthy();
    }
  });
});
