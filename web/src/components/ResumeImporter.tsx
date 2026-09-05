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

export function ResumeImporter({
  apiBaseUrl,
  onImported,
}: {
  apiBaseUrl: string;
  onImported: (result: ResumeImportResult) => void;
}) {
  const [resumes, setResumes] = useState<ResumeView[]>([]);
  const [roles, setRoles] = useState<TargetRoleView[]>([]);
  const [destination, setDestination] = useState("new");
  const [name, setName] = useState("");
  const [roleId, setRoleId] = useState("__new__");
  const [newRole, setNewRole] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputId = useId();

  useEffect(() => {
    const request = new AbortController();
    void Promise.all([
      fetchResumes({ apiBaseUrl, signal: request.signal }),
      fetchTargetRoles({ apiBaseUrl, signal: request.signal }),
    ]).then(([nextResumes, nextRoles]) => {
      setResumes(nextResumes);
      setRoles(nextRoles);
      setRoleId(nextRoles.length > 0 ? nextRoles[0].id : "__new__");
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
          ? { file, name: name.trim(), targetRoleId: selectedRoleId }
          : { file, resumeId: destination },
        { apiBaseUrl },
      );
      onImported(result);
      setFile(null);
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
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <label className="resume-file-picker" htmlFor={fileInputId}>
          <span>选择文件</span>
          <small>{file?.name ?? "PDF、TXT 或 Markdown"}</small>
        </label>
      </div>
      {error ? <div className="error-banner" role="alert">{error}</div> : null}
      <button type="submit" disabled={!valid || pending}>
        {pending ? "导入中…" : "导入简历"}
      </button>
      <small>支持 PDF、TXT、Markdown，最大 5 MiB。文件内容只进入本地简历存储。</small>
    </form>
  );
}
