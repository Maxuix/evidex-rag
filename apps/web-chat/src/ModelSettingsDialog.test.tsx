import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ModelSettingsDialog } from "./ModelSettingsDialog";
import type { ApiClient } from "./api/client";
import type { ModelSettings } from "./api/types";

const NOW = "2026-09-02T00:00:00Z";

function settings(): ModelSettings {
  return {
    providers: [{
      id: "provider-a",
      revision_id: "provider-a-r1",
      revision: 1,
      name: "OpenCode Go",
      protocol: "openai_compatible",
      base_url: "https://provider.invalid/v1",
      timeout_seconds: 60,
      max_retries: 3,
      max_concurrency: 3,
      enabled: true,
      api_key_configured: true,
      configuration_fingerprint: "provider-fingerprint",
      created_at: NOW,
      updated_at: NOW,
    }],
    profiles: [],
    selection: {
      chat_profile_revision_id: null,
      text_embedding_profile_revision_id: null,
      multimodal_embedding_profile_revision_id: null,
      updated_at: NOW,
    },
  };
}

afterEach(() => cleanup());

describe("model settings create forms", () => {
  it("resets and closes the Chat form after a successful create", async () => {
    const current = settings();
    const client = {
      createModelProfile: vi.fn().mockResolvedValue({}),
      getModelSettings: vi.fn().mockResolvedValue(current),
      listProviderModels: vi.fn().mockResolvedValue({ models: ["mimo-v2.5"] }),
    } as unknown as ApiClient;
    const user = userEvent.setup();

    render(
      <ModelSettingsDialog
        client={client}
        initial={current}
        onChange={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const summary = screen.getByText("＋ 添加 Chat 模型");
    const details = summary.closest("details");
    expect(details).not.toBeNull();
    await user.click(summary);
    expect(details).toHaveProperty("open", true);

    await user.selectOptions(screen.getByLabelText("提供商"), "provider-a");
    await user.type(screen.getByLabelText("显示名称"), "mimo-v2.5");
    await user.type(screen.getByLabelText("模型 ID"), "mimo-v2.5");
    await user.click(screen.getByRole("button", { name: "保存模型" }));

    await waitFor(() => expect(client.createModelProfile).toHaveBeenCalledTimes(1));
    await waitFor(() => {
      const refreshed = screen.getByText("＋ 添加 Chat 模型").closest("details");
      expect(refreshed).toHaveProperty("open", false);
    });
  });
});
