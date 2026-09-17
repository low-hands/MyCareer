import { resumeDocumentUrl, type ResumeJobMatchView } from "../api/client";
import { SavedJobCard } from "./SavedJobCard";

function time(value: string | null): string {
  return value ? new Date(value).toLocaleString("zh-CN") : "历史记录未保存";
}

export function MatchProvenance({
  match, apiBaseUrl,
}: { match: ResumeJobMatchView; apiBaseUrl: string }) {
  return (
    <section className="match-provenance" aria-label="匹配报告版本信息">
      <h3>参与匹配的版本</h3>
      <dl>
        <dt>岗位</dt><dd>{match.company_name ?? "公司信息不可用"} · {match.job_title ?? "岗位信息不可用"}</dd>
        <dt>岗位 ID</dt><dd>{match.job_posting_id}</dd>
        <dt>JD 版本</dt><dd>{match.jd_version === null ? "历史版本号未保存" : `v${match.jd_version}`}{match.current_jd === false ? "（历史快照，岗位已有新版本）" : ""}</dd>
        <dt>JD 快照 ID</dt><dd>{match.jd_snapshot_id}</dd>
        <dt>JD 抓取时间</dt><dd>{time(match.jd_captured_at)}</dd>
        <dt>简历</dt><dd>{match.resume_name ?? "简历名称未保存"}</dd>
        <dt>简历 ID</dt><dd>{match.resume_id ?? "历史记录未保存"}</dd>
        <dt>简历版本</dt><dd>{match.resume_version_number === null ? "历史版本号未保存" : `v${match.resume_version_number}`}</dd>
        <dt>简历版本 ID</dt><dd>{match.resume_version_id}</dd>
        <dt>版本创建时间</dt><dd>{time(match.resume_created_at)}</dd>
        <dt>报告 ID</dt><dd>{match.report_id}</dd>
        <dt>报告生成时间</dt><dd>{time(match.created_at)}</dd>
        <dt>匹配器版本</dt><dd>{match.matcher_version}</dd>
      </dl>
      {match.resume_available && match.resume_id ? (
        <a href={resumeDocumentUrl(match.resume_id, match.resume_version_id, { apiBaseUrl })} target="_blank" rel="noreferrer">
          查看参与匹配的简历原文件
        </a>
      ) : <p role="status">原简历已删除或不可访问，无法查看原文件；报告与版本引用仍保留。</p>}
      {match.jd_available ? (
        <SavedJobCard
          apiBaseUrl={apiBaseUrl}
          resource={{
            kind: "saved_job",
            resourceId: match.jd_snapshot_id,
            title: `参与匹配的 JD v${match.jd_version ?? "未知"}`,
          }}
        />
      ) : <p role="status">原 JD 已删除或不可访问，无法查看原文；报告与版本引用仍保留。</p>}
    </section>
  );
}
