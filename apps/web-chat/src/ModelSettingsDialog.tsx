import { FormEvent, useEffect, useMemo, useState } from "react";

import type { ApiClient } from "./api/client";
import type {
  ModelKind,
  ModelProfile,
  ModelProvider,
  ModelProviderProtocol,
  ModelSettings,
} from "./api/types";

type SettingsSection = "providers" | "chat" | "embedding";
type ProviderMutation = {
  name: string;
  protocol: ModelProviderProtocol;
  base_url: string;
  api_key?: string;
  timeout_seconds: number;
  max_retries: number;
  max_concurrency: number;
  enabled?: boolean;
};
type ProfileMutation = {
  provider_id: string;
  name: string;
  kind: ModelKind;
  model: string;
  parameters: Record<string, unknown>;
  enabled?: boolean;
};

const CHAT_DEFAULTS = {
  temperature: 0.2,
  top_p: 0.9,
  sampling_top_k: 40,
  max_output_tokens: 4096,
  reasoning_effort: "off",
  structured_output_mode: "json_object",
  vision_enabled: false,
} as const;

export function ModelSettingsDialog({
  client,
  initial,
  onChange,
  onClose,
}: {
  client: ApiClient;
  initial: ModelSettings | null;
  onChange: (value: ModelSettings) => void;
  onClose: () => void;
}) {
  const [settings, setSettings] = useState<ModelSettings | null>(initial);
  const [section, setSection] = useState<SettingsSection>("chat");
  const [loading, setLoading] = useState(!initial);
  const [busy, setBusy] = useState(false);
  const [catalogBusy, setCatalogBusy] = useState<string | null>(null);
  const [catalogs, setCatalogs] = useState<Record<string, string[]>>({});
  const [editingProvider, setEditingProvider] = useState<ModelProvider | null>(null);
  const [editingProfile, setEditingProfile] = useState<ModelProfile | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const value = await client.getModelSettings();
      setSettings(value);
      onChange(value);
    } catch (caught) {
      setError(message(caught));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!initial) void refresh();
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);

  const run = async (operation: () => Promise<unknown>, onSuccess?: () => void) => {
    setBusy(true);
    setError(null);
    try {
      await operation();
      await refresh();
      onSuccess?.();
    } catch (caught) {
      setError(message(caught));
    } finally {
      setBusy(false);
    }
  };

  const loadCatalog = async (providerId: string) => {
    const provider = settings?.providers.find((item) => item.id === providerId);
    if (!provider || provider.protocol !== "openai_compatible" || !provider.enabled) return;
    setCatalogBusy(providerId);
    setError(null);
    try {
      const value = await client.listProviderModels(providerId);
      setCatalogs((current) => ({ ...current, [providerId]: value.models }));
    } catch (caught) {
      setError(message(caught));
    } finally {
      setCatalogBusy(null);
    }
  };

  return (
    <div className="settings-backdrop" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="model-settings-title">
        <header>
          <div>
            <h2 id="model-settings-title">模型设置</h2>
            <p>配置保存在本机；API Key 不会返回浏览器。</p>
          </div>
          <button className="icon-button" type="button" aria-label="关闭模型设置" onClick={onClose}>×</button>
        </header>

        <nav className="settings-tabs" aria-label="模型设置分类">
          {([
            ["chat", "Chat 模型"],
            ["embedding", "Embedding 模型"],
            ["providers", "提供商"],
          ] as const).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={section === value ? "active" : ""}
              aria-current={section === value ? "page" : undefined}
              onClick={() => {
                setSection(value);
                setEditingProfile(null);
                setEditingProvider(null);
              }}
            >
              {label}
            </button>
          ))}
        </nav>

        {error ? <div className="settings-error">{error}</div> : null}
        {loading && !settings ? <div className="settings-loading">正在读取模型配置…</div> : null}
        {settings ? (
          <div className="settings-content">
            {section === "providers" ? (
              <ProviderPanel
                settings={settings}
                disabled={busy}
                catalogs={catalogs}
                catalogBusy={catalogBusy}
                editing={editingProvider}
                onEdit={setEditingProvider}
                onLoadCatalog={loadCatalog}
                onCreate={(payload) => run(
                  () => client.createModelProvider(requiredProviderPayload(payload)),
                )}
                onUpdate={(providerId, payload) => run(
                  () => client.updateModelProvider(providerId, payload),
                  () => setEditingProvider(null),
                )}
              />
            ) : (
              <ModelPanel
                category={section}
                settings={settings}
                disabled={busy}
                catalogs={catalogs}
                catalogBusy={catalogBusy}
                editing={editingProfile}
                onEdit={setEditingProfile}
                onLoadCatalog={loadCatalog}
                onValidate={(profileId) => run(() => client.validateModelProfile(profileId))}
                onSaveDefaults={(selection) => run(() => client.updateModelSelection(selection))}
                onCreate={(payload) => run(() => client.createModelProfile(payload))}
                onUpdate={(profileId, payload) => run(
                  () => client.updateModelProfile(profileId, {
                    provider_id: payload.provider_id,
                    name: payload.name,
                    model: payload.model,
                    parameters: payload.parameters,
                    enabled: payload.enabled,
                  }),
                  () => setEditingProfile(null),
                )}
              />
            )}
          </div>
        ) : null}
      </section>
    </div>
  );
}

function ProviderPanel({
  settings,
  disabled,
  catalogs,
  catalogBusy,
  editing,
  onEdit,
  onLoadCatalog,
  onCreate,
  onUpdate,
}: {
  settings: ModelSettings;
  disabled: boolean;
  catalogs: Record<string, string[]>;
  catalogBusy: string | null;
  editing: ModelProvider | null;
  onEdit: (value: ModelProvider | null) => void;
  onLoadCatalog: (providerId: string) => void;
  onCreate: (value: ProviderMutation) => void;
  onUpdate: (providerId: string, value: ProviderMutation) => void;
}) {
  return (
    <section className="settings-panel">
      <div className="settings-panel-heading">
        <div><h3>模型提供商</h3><p>先添加连接，再到 Chat 或 Embedding 区域选择模型。</p></div>
      </div>
      <div className="settings-cards provider-cards">
        {settings.providers.map((provider) => (
          <article className="settings-card" key={provider.id}>
            <div className="settings-card-heading">
              <strong>{provider.name}</strong>
              <span className={provider.enabled ? "provider-status enabled" : "provider-status"}>{provider.enabled ? "启用" : "停用"}</span>
            </div>
            <span>{provider.protocol === "openai_compatible" ? "OpenAI 兼容" : "通义多模态"}</span>
            <code title={provider.base_url}>{provider.base_url}</code>
            <small>密钥已配置 · 修订 {provider.revision}</small>
            {catalogs[provider.id] ? <small>已获取 {catalogs[provider.id].length} 个模型</small> : null}
            <div className="settings-card-actions">
              <button type="button" disabled={disabled} onClick={() => onEdit(provider)}>修改</button>
              {provider.protocol === "openai_compatible" ? (
                <button type="button" disabled={disabled || !provider.enabled || catalogBusy === provider.id} onClick={() => onLoadCatalog(provider.id)}>
                  {catalogBusy === provider.id ? "获取中…" : "获取模型"}
                </button>
              ) : null}
            </div>
          </article>
        ))}
        {!settings.providers.length ? <p className="settings-empty">尚未添加提供商。</p> : null}
      </div>

      {editing ? (
        <ProviderForm
          key={editing.revision_id}
          disabled={disabled}
          initial={editing}
          onCancel={() => onEdit(null)}
          onSubmit={(payload) => onUpdate(editing.id, payload)}
        />
      ) : null}
      <ProviderForm disabled={disabled} onSubmit={onCreate} />
    </section>
  );
}

function ProviderForm({
  disabled,
  initial,
  onSubmit,
  onCancel,
}: {
  disabled: boolean;
  initial?: ModelProvider;
  onSubmit: (value: ProviderMutation) => void;
  onCancel?: () => void;
}) {
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const apiKey = String(data.get("api_key") ?? "").trim();
    onSubmit({
      name: String(data.get("name")),
      protocol: String(data.get("protocol")) as ModelProviderProtocol,
      base_url: String(data.get("base_url")),
      ...(apiKey ? { api_key: apiKey } : {}),
      timeout_seconds: Number(data.get("timeout_seconds")),
      max_retries: Number(data.get("max_retries")),
      max_concurrency: Number(data.get("max_concurrency")),
      ...(initial ? { enabled: data.get("enabled") === "on" } : {}),
    });
  };
  return (
    <details className={`settings-form ${initial ? "editing" : ""}`} open={Boolean(initial)}>
      <summary>{initial ? `修改提供商 · ${initial.name}` : "＋ 添加提供商"}</summary>
      <form onSubmit={submit}>
        <div className="settings-field-row two-columns">
          <label className="settings-field"><span>名称</span><input required name="name" maxLength={255} defaultValue={initial?.name} placeholder="例如：本地兼容服务" /></label>
          <label className="settings-field"><span>协议</span><select name="protocol" defaultValue={initial?.protocol ?? "openai_compatible"}><option value="openai_compatible">OpenAI 兼容</option><option value="tongyi_multimodal">通义多模态 Embedding</option></select></label>
        </div>
        <label className="settings-field"><span>Base URL / Endpoint</span><input required name="base_url" type="url" defaultValue={initial?.base_url} placeholder="http://127.0.0.1:11434/v1" /></label>
        <label className="settings-field"><span>API Key</span><input required={!initial} name="api_key" type="password" autoComplete="new-password" placeholder={initial ? "留空表示不更换" : undefined} /></label>
        <div className="settings-field-row">
          <label className="settings-field"><span>超时（秒）</span><input required name="timeout_seconds" type="number" min="1" max="600" defaultValue={initial?.timeout_seconds ?? 30} /></label>
          <label className="settings-field"><span>重试</span><input required name="max_retries" type="number" min="0" max="10" defaultValue={initial?.max_retries ?? 2} /></label>
          <label className="settings-field"><span>并发</span><input required name="max_concurrency" type="number" min="1" max="32" defaultValue={initial?.max_concurrency ?? 2} /></label>
        </div>
        {initial ? <label className="settings-check"><input name="enabled" type="checkbox" defaultChecked={initial.enabled} />启用此提供商</label> : null}
        <div className="settings-form-actions">
          <button className="primary-button" type="submit" disabled={disabled}>{initial ? "保存修改" : "保存提供商"}</button>
          {onCancel ? <button type="button" disabled={disabled} onClick={onCancel}>取消</button> : null}
        </div>
      </form>
    </details>
  );
}

function ModelPanel({
  category,
  settings,
  disabled,
  catalogs,
  catalogBusy,
  editing,
  onEdit,
  onLoadCatalog,
  onValidate,
  onSaveDefaults,
  onCreate,
  onUpdate,
}: {
  category: "chat" | "embedding";
  settings: ModelSettings;
  disabled: boolean;
  catalogs: Record<string, string[]>;
  catalogBusy: string | null;
  editing: ModelProfile | null;
  onEdit: (value: ModelProfile | null) => void;
  onLoadCatalog: (providerId: string) => void;
  onValidate: (profileId: string) => void;
  onSaveDefaults: (value: ModelSettings["selection"]) => void;
  onCreate: (value: ProfileMutation) => void;
  onUpdate: (profileId: string, value: ProfileMutation) => void;
}) {
  const kinds: ModelKind[] = category === "chat"
    ? ["chat"]
    : ["text_embedding", "multimodal_embedding"];
  const profiles = settings.profiles.filter((profile) => kinds.includes(profile.kind));
  const selectedEdit = editing && kinds.includes(editing.kind) ? editing : null;
  return (
    <section className="settings-panel">
      <div className="settings-panel-heading">
        <div>
          <h3>{category === "chat" ? "Chat 模型" : "Embedding 模型"}</h3>
          <p>{category === "chat" ? "控制回答生成、采样和思考参数。" : "文本与多模态向量模型独立配置。"}</p>
        </div>
      </div>
      <DefaultModels category={category} settings={settings} disabled={disabled} onSave={onSaveDefaults} />
      <div className="settings-cards model-cards">
        {profiles.map((profile) => (
          <article className="settings-card model-card" key={profile.id}>
            <div className="settings-card-heading">
              <strong>{profile.name}</strong>
              <div className={`validation-status ${profile.validation_status}`}>{validationLabel(profile.validation_status)}</div>
            </div>
            <span>{kindLabel(profile.kind)} · {profile.model}</span>
            <small>{parameterSummary(profile)}</small>
            <small>修订 {profile.revision}{profile.enabled ? "" : " · 已停用"}</small>
            <div className="settings-card-actions">
              <button type="button" disabled={disabled} onClick={() => onEdit(profile)}>修改</button>
              <button type="button" disabled={disabled || !profile.enabled} onClick={() => onValidate(profile.id)}>验证连接</button>
            </div>
          </article>
        ))}
        {!profiles.length ? <p className="settings-empty">尚未添加{category === "chat" ? " Chat" : " Embedding"} 模型。</p> : null}
      </div>

      {selectedEdit ? (
        <ProfileForm
          key={selectedEdit.revision_id}
          settings={settings}
          disabled={disabled}
          allowedKinds={[selectedEdit.kind]}
          initial={selectedEdit}
          catalogs={catalogs}
          catalogBusy={catalogBusy}
          onLoadCatalog={onLoadCatalog}
          onCancel={() => onEdit(null)}
          onSubmit={(payload) => onUpdate(selectedEdit.id, payload)}
        />
      ) : null}
      <ProfileForm
        key={`create-${category}`}
        settings={settings}
        disabled={disabled || !settings.providers.length}
        allowedKinds={kinds}
        catalogs={catalogs}
        catalogBusy={catalogBusy}
        onLoadCatalog={onLoadCatalog}
        onSubmit={onCreate}
      />
    </section>
  );
}

function DefaultModels({ category, settings, disabled, onSave }: {
  category: "chat" | "embedding";
  settings: ModelSettings;
  disabled: boolean;
  onSave: (value: ModelSettings["selection"]) => void;
}) {
  const [selection, setSelection] = useState(settings.selection);
  useEffect(() => setSelection(settings.selection), [settings.selection]);
  const entries = category === "chat"
    ? [["chat_profile_revision_id", "默认 Chat", "chat"]] as const
    : [
      ["text_embedding_profile_revision_id", "默认文本 Embedding", "text_embedding"],
      ["multimodal_embedding_profile_revision_id", "默认多模态 Embedding", "multimodal_embedding"],
    ] as const;
  const eligible = (kind: ModelKind) => settings.profiles.filter(
    (profile) => profile.kind === kind && profile.enabled && profile.validation_status === "valid",
  );
  return (
    <div className={`default-models ${category}`}>
      {entries.map(([field, label, kind]) => {
        const options = eligible(kind);
        const selectedIsHistorical = Boolean(selection[field])
          && !options.some((profile) => profile.revision_id === selection[field]);
        return (
          <label className="settings-field" key={field}>
            <span>{label}</span>
            <select value={selection[field] ?? ""} disabled={disabled} onChange={(event) => setSelection({ ...selection, [field]: event.target.value || null })}>
              <option value="">未选择</option>
              {selectedIsHistorical ? <option value={selection[field] ?? ""}>当前历史修订</option> : null}
              {options.map((profile) => <option key={profile.revision_id} value={profile.revision_id}>{profile.name} · r{profile.revision}</option>)}
            </select>
          </label>
        );
      })}
      <button className="primary-button" type="button" disabled={disabled} onClick={() => onSave(selection)}>保存默认模型</button>
    </div>
  );
}

function ProfileForm({
  settings,
  disabled,
  allowedKinds,
  initial,
  catalogs,
  catalogBusy,
  onLoadCatalog,
  onSubmit,
  onCancel,
}: {
  settings: ModelSettings;
  disabled: boolean;
  allowedKinds: ModelKind[];
  initial?: ModelProfile;
  catalogs: Record<string, string[]>;
  catalogBusy: string | null;
  onLoadCatalog: (providerId: string) => void;
  onSubmit: (value: ProfileMutation) => void;
  onCancel?: () => void;
}) {
  const [kind, setKind] = useState<ModelKind>(initial?.kind ?? allowedKinds[0]);
  const [providerId, setProviderId] = useState(initial?.provider_id ?? "");
  const providers = useMemo(() => settings.providers.filter((provider) => (
    provider.enabled && (kind === "multimodal_embedding"
      ? provider.protocol === "tongyi_multimodal"
      : provider.protocol === "openai_compatible")
  )), [kind, settings.providers]);
  const provider = settings.providers.find((item) => item.id === providerId);
  const models = catalogs[providerId] ?? [];
  const chatParameters = initial?.parameters.type === "chat" ? initial.parameters : CHAT_DEFAULTS;
  const embeddingParameters = initial?.parameters.type === "embedding" ? initial.parameters : null;

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const parameters = kind === "chat" ? {
      type: "chat",
      temperature: Number(data.get("temperature")),
      top_p: optionalNumber(data.get("top_p")),
      sampling_top_k: optionalNumber(data.get("sampling_top_k")),
      max_output_tokens: Number(data.get("max_output_tokens")),
      reasoning_effort: String(data.get("reasoning_effort")),
      structured_output_mode: String(data.get("structured_output_mode")),
      vision_enabled: data.get("vision_enabled") === "on",
    } : {
      type: "embedding",
      dimension: kind === "text_embedding" ? 1024 : 768,
      max_batch_size: Number(data.get("max_batch_size")),
      distance_metric: "cosine",
      vector_data_type: "float32",
      normalization: "l2",
    };
    const model = String(data.get("model"));
    const displayName = String(data.get("name")).trim();
    onSubmit({
      provider_id: String(data.get("provider_id")),
      name: displayName || model,
      kind,
      model,
      parameters,
      ...(initial ? { enabled: data.get("enabled") === "on" } : {}),
    });
  };

  return (
    <details className={`settings-form ${initial ? "editing" : ""}`} open={Boolean(initial)}>
      <summary>{initial ? `修改模型 · ${initial.name}` : `＋ 添加${allowedKinds.length === 1 && allowedKinds[0] === "chat" ? " Chat" : " Embedding"} 模型`}</summary>
      <form onSubmit={submit}>
        {allowedKinds.length > 1 ? (
          <label className="settings-field"><span>类型</span><select value={kind} onChange={(event) => {
            setKind(event.target.value as ModelKind);
            setProviderId("");
          }}><option value="text_embedding">文本 Embedding（1024 维）</option><option value="multimodal_embedding">多模态 Embedding（768 维）</option></select></label>
        ) : <div className="settings-form-kind">{kindLabel(kind)}</div>}
        <div className="settings-field-row two-columns">
          <label className="settings-field"><span>提供商</span><select required name="provider_id" value={providerId} onChange={(event) => {
            const value = event.target.value;
            setProviderId(value);
            if (value && !catalogs[value]) void onLoadCatalog(value);
          }}><option value="">请选择</option>{providers.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
          <label className="settings-field"><span>显示名称</span><input name="name" maxLength={255} defaultValue={initial?.name} placeholder="留空则使用模型 ID" /></label>
        </div>
        <div className="model-picker">
          <label className="settings-field"><span>模型 ID</span><input required name="model" list={`model-catalog-${kind}-${initial?.id ?? "new"}`} maxLength={255} defaultValue={initial?.model} placeholder={models.length ? "选择或输入模型" : "例如：qwen-plus"} /></label>
          <datalist id={`model-catalog-${kind}-${initial?.id ?? "new"}`}>{models.map((model) => <option key={model} value={model} />)}</datalist>
          {provider?.protocol === "openai_compatible" ? (
            <button type="button" disabled={disabled || catalogBusy === providerId} onClick={() => onLoadCatalog(providerId)}>{catalogBusy === providerId ? "获取中…" : "刷新模型列表"}</button>
          ) : null}
        </div>
        {providerId ? <small className="model-catalog-note">{models.length ? `可选择 ${models.length} 个已发现模型，也可手动输入。` : "该提供商暂无可用模型列表，可手动输入模型 ID。"}</small> : null}
        {kind === "chat" ? <>
          <div className="settings-field-row">
            <label className="settings-field"><span>温度</span><input required name="temperature" type="number" min="0" max="2" step="0.05" defaultValue={chatParameters.temperature} /></label>
            <label className="settings-field"><span>Top P</span><input name="top_p" type="number" min="0.01" max="1" step="0.01" defaultValue={chatParameters.top_p ?? ""} /></label>
            <label className="settings-field"><span>采样 Top K</span><input name="sampling_top_k" type="number" min="1" max="1000" defaultValue={chatParameters.sampling_top_k ?? ""} /></label>
          </div>
          <div className="settings-field-row">
            <label className="settings-field"><span>最大输出 Token</span><input required name="max_output_tokens" type="number" min="1" max="8192" defaultValue={chatParameters.max_output_tokens} /></label>
            <label className="settings-field"><span>思考程度</span><select name="reasoning_effort" defaultValue={chatParameters.reasoning_effort}><option value="off">关闭</option><option value="low">低</option><option value="medium">中</option><option value="high">高</option></select></label>
            <label className="settings-field"><span>结构化输出</span><select name="structured_output_mode" defaultValue={chatParameters.structured_output_mode}><option value="json_object">JSON Object</option><option value="json_schema">JSON Schema</option></select></label>
          </div>
          <label className="settings-check"><input name="vision_enabled" type="checkbox" defaultChecked={chatParameters.vision_enabled} />允许视觉输入</label>
        </> : <label className="settings-field"><span>最大批量</span><input required name="max_batch_size" type="number" min="1" max="100" defaultValue={embeddingParameters?.max_batch_size ?? 10} /></label>}
        {initial ? <label className="settings-check"><input name="enabled" type="checkbox" defaultChecked={initial.enabled} />启用此模型</label> : null}
        <div className="settings-form-actions">
          <button className="primary-button" type="submit" disabled={disabled || !providers.length}>{initial ? "保存修改" : "保存模型"}</button>
          {onCancel ? <button type="button" disabled={disabled} onClick={onCancel}>取消</button> : null}
        </div>
      </form>
    </details>
  );
}

function requiredProviderPayload(value: ProviderMutation): ProviderMutation & { api_key: string } {
  if (!value.api_key) throw new Error("请填写 API Key。");
  return { ...value, api_key: value.api_key };
}

function optionalNumber(value: FormDataEntryValue | null): number | null {
  return typeof value === "string" && value.trim() ? Number(value) : null;
}

function kindLabel(kind: ModelKind): string {
  return kind === "chat" ? "Chat" : kind === "text_embedding" ? "文本 Embedding" : "多模态 Embedding";
}

function validationLabel(status: string): string {
  return status === "valid" ? "验证成功" : status === "invalid" ? "验证失败" : "尚未验证";
}

function parameterSummary(profile: ModelProfile): string {
  const value = profile.parameters;
  if (value.type === "embedding") return `${value.dimension} 维 · 批量 ${value.max_batch_size}`;
  return `温度 ${value.temperature} · Top P ${value.top_p ?? "关闭"} · Top K ${value.sampling_top_k ?? "关闭"} · 输出 ${value.max_output_tokens}`;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : "模型设置操作失败。";
}
