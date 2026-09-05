import { useState } from "react";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { KnowledgeBase } from "./api/types";
import { KnowledgeScopePicker } from "./KnowledgeScopePicker";

const libraries = [
  { id: "enterprise", name: "企业知识库", description: "制度与业务资料" },
  { id: "operations", name: "运维手册", description: "备份和数据库" },
  { id: "long", name: "formal-agent-v6-e2e-20260905-121954", description: "验收记录" },
] as KnowledgeBase[];

afterEach(cleanup);

function Fixture({ saving = false, onChange = vi.fn() }) {
  const [selected, setSelected] = useState(["enterprise"]);
  return <><KnowledgeScopePicker knowledgeBases={libraries} selectedIds={selected} disabled={false}
    saving={saving} loading={false} hasMore={false} activeRun={true}
    onChange={async ids => { onChange(ids); setSelected(ids); }} onSelectAll={vi.fn()} onLoadMore={vi.fn()} />
    <textarea aria-label="问题" /></>;
}

describe("compact knowledge scope picker", () => {
  it("starts collapsed, filters by description and preserves selections hidden by search", async () => {
    const onChange = vi.fn(); render(<Fixture onChange={onChange} />);
    expect(screen.queryByRole("checkbox")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "搜索范围：企业知识库" }));
    expect(document.activeElement).toBe(screen.getByRole("searchbox"));
    await userEvent.type(screen.getByRole("searchbox"), "数据库");
    expect(screen.queryByRole("checkbox", { name: "企业知识库" })).toBeNull();
    await userEvent.click(screen.getByRole("checkbox", { name: "运维手册" }));
    expect(onChange).toHaveBeenCalledWith(["enterprise", "operations"]);
    expect(screen.getByRole("status").textContent).toBe("已选 2 个");
    await userEvent.click(screen.getByRole("button", { name: "完成" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("button", { name: "搜索范围：企业知识库、运维手册" }).textContent).toContain("+1");
  });

  it("closes with Escape, outside click or keyboard departure without losing selection", async () => {
    render(<Fixture />);
    const trigger = screen.getByRole("button", { name: "搜索范围：企业知识库" });
    await userEvent.click(trigger);
    await userEvent.keyboard("{Escape}");
    expect(document.activeElement).toBe(trigger);
    expect(screen.queryByRole("dialog")).toBeNull();
    await userEvent.click(trigger);
    await userEvent.click(screen.getByRole("textbox", { name: "问题" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    await userEvent.click(trigger);
    const done = screen.getByRole("button", { name: "完成" }); done.focus();
    await userEvent.tab();
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("button", { name: "搜索范围：企业知识库" })).toBeTruthy();
  });

  it("keeps search available while saving and prevents conflicting scope writes", async () => {
    const onChange = vi.fn(); render(<Fixture saving onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "搜索范围：企业知识库" }));
    expect(screen.getByRole("status").textContent).toBe("保存中…");
    await userEvent.click(screen.getByRole("checkbox", { name: "运维手册" }));
    expect(onChange).not.toHaveBeenCalled();
    await userEvent.type(screen.getByRole("searchbox"), "找不到的库");
    expect(screen.getByText("没有找到匹配的知识库")).toBeTruthy();
    expect(screen.getByText("修改用于下一轮，当前回答范围不变")).toBeTruthy();
  });
});
