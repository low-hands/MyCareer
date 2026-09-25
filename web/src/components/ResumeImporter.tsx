import { FormEvent, useEffect, useId, useState } from "react";

import {
  createTargetRole,
  fetchResumes,
  fetchTargetRoles,
  importResume,
  type ResumeImportResult,
  type ResumeView,
  type TargetRoleView,
} from "../api/client";

/**
 * The one import form, used by the resume library and by the chat composer.
 *
 * Both paths call `POST /v1/resumes/import`, so a file dropped into chat ends
 * up in the same library, as the same kind of version, as one imported from
 * the library page. The chat passes the dropped file as `initialFile` and
 * gets the created version back through `onImported` to attach to its
 * message; nothing about the file itself is kept on the client.
 */
export function ResumeImporter({
  apiBaseUrl,
  onImported,
  initialFile = null,
  onCancel,
  submitLabel = "导入简历",
  initialTargetRoleId,
}: {
  apiBaseUrl: string;
  onImported: (result: ResumeImportResult) => void;
  initialFile?: File | null;
  onCancel?: () => void;
  submitLabel?: string;
  /** Preselect this target role for a new resume, as "import into this role" does. */
  initialTargetRoleId?: string;
}) {
  const [resumes, setResumes] = useState<ResumeView[]>([]);
  const [roles, setRoles] = useState<TargetRoleView[]>([]);
  const [destination, setDestination] = useState("new");
  const [name, setName] = useState(() => suggestedName(initialFile));
  const [roleId, setRoleId] = useState("__new__");
  const [newRole, setNewRole] = useState("");
  const [file, setFile] = useState<File | null>(initialFile);
  // One key per chosen file: a retry after a failed or interrupted submit
  // reuses it, so the server returns the version it already created instead
  // of adding a duplicate. A new file gets a new key.
  const [idempotencyKey, setIdempotencyKey] = useState(() => crypto.randomUUID());
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputId = useId();

  function chooseFile(next: File | null): void {
    setFile(next);
    setIdempotencyKey(crypto.randomUUID());
    if (next && !name.trim()) setName(suggestedName(next));
  }

  useEffect(() => {
    const request = new AbortController();
    void Promise.all([
      fetchResumes({ apiBaseUrl, signal: request.signal }),
      fetchTargetRoles({ apiBaseUrl, signal: request.signal }),
    ]).then(([nextResumes, nextRoles]) => {
      setResumes(nextResumes);
      setRoles(nextRoles);
      setRoleId(
        initialTargetRoleId && nextRoles.some((role) => role.id === initialTargetRoleId)
          ? initialTargetRoleId
          : nextRoles.length > 0 ? nextRoles[0].id : "__new__",
      );
    }).catch((cause: unknown) => {
      if (!request.signal.aborted) {
        setError(cause instanceof Error ? cause.message : "读取简历分类失败。");
      }
    });
    return () => request.abort();
  }, [apiBaseUrl]);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!file || (destination === "new" && !name.trim())) return;
    setPending(true);
    setError(null);
    try {
      let selectedRoleId = roleId;
      if (destination === "new" && roleId === "__new__") {
        selectedRoleId = (await createTargetRole(newRole, { apiBaseUrl })).id;
      }
      const result = await importResume(
        destination === "new"
          ? { file, name: name.trim(), targetRoleId: selectedRoleId, idempotencyKey }
          : { file, resumeId: destination, idempotencyKey },
        { apiBaseUrl },
      );
      onImported(result);
      setFile(null);
      setIdempotencyKey(crypto.randomUUID());
      setName("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "导入简历失败。");
    } finally {
      setPending(false);
    }
  }

  const valid = Boolean(
    file &&
    (destination !== "new" ||
      (name.trim() && roleId && (roleId !== "__new__" || newRole.trim()))),
  );

  return (
    <form className="resume-importer" onSubmit={(event) => void submit(event)}>
      <label>
        导入位置
        <select value={destination} onChange={(event) => setDestination(event.target.value)}>
          <option value="new">创建新简历</option>
          {resumes.map((resume) => (
            <option value={resume.id} key={resume.id}>追加到 {resume.name}</option>
          ))}
        </select>
      </label>
      {destination === "new" ? (
        <>
          <label>
            简历名称
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="例如：AI 产品经理简历" />
          </label>
          <label>
            目标岗位
            <select value={roleId} onChange={(event) => setRoleId(event.target.value)}>
              {roles.map((role) => (
                <option value={role.id} key={role.id}>{role.title}</option>
              ))}
              <option value="__new__">新建目标岗位…</option>
            </select>
          </label>
          {roleId === "__new__" ? (
            <label>
              新目标岗位
              <input value={newRole} onChange={(event) => setNewRole(event.target.value)} placeholder="例如：AI 产品经理" />
            </label>
          ) : null}
        </>
      ) : null}
      <div className="resume-file-control">
        <span className="resume-file-label">文件</span>
        <input
          id={fileInputId}
          className="resume-file-input"
          type="file"
          accept=".pdf,.txt,.md,.markdown"
          onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
        />
        <label className="resume-file-picker" htmlFor={fileInputId}>
          <span>选择文件</span>
          <small>{file?.name ?? "PDF、TXT 或 Markdown"}</small>
        </label>
      </div>
      {error ? <div className="error-banner" role="alert">{error}</div> : null}
      <div className="resume-importer-actions">
        <button type="submit" disabled={!valid || pending}>
          {pending ? "导入中…" : submitLabel}
        </button>
        {onCancel ? (
          <button type="button" className="soft-button" onClick={onCancel} disabled={pending}>
            取消
          </button>
        ) : null}
      </div>
      <small>
        支持 PDF、TXT、Markdown，最大 5 MiB。“上传”只把原文件保存到本地简历库；
        只有“发送消息 / 让 Agent 分析”时，提取出的简历内容才会发给配置的模型供应商。
      </small>
    </form>
  );
}

function suggestedName(file: File | null): string {
  if (!file) return "";
  return file.name.replace(/\.[^.]+$/, "").trim().slice(0, 80);
}
