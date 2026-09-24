import { resumeDocumentUrl, type ResumeJobMatchView } from "../api/client";
import { SavedJobCard } from "./SavedJobCard";

function time(value: string | null): string {
  return value ? new Date(value).toLocaleString("zh-CN", { dateStyle: "medium", timeStyle: "short" }) : "时间未保存";
}

export function MatchProvenance({
  match, apiBaseUrl,
}: { match: ResumeJobMatchView; apiBaseUrl: string }) {
  return (
    <section className="match-provenance" aria-label="匹配报告使用的内容">
      <div className="match-provenance-heading">
        <div>
          <span className="card-kicker">参考内容</span>
          <h3>这份结果参考了</h3>
        </div>
        <time>{time(match.created_at)} 生成</time>
      </div>
      <div className="match-source-grid">
        <div className="match-source-card">
          <span className="match-source-icon" aria-hidden="true">JD</span>
          <div>
            <small>目标岗位</small>
            <strong>{match.company_name ?? "公司信息不可用"} · {match.job_title ?? "岗位信息不可用"}</strong>
            <span>{match.jd_version === null ? "版本未记录" : `JD v${match.jd_version}`}{match.current_jd === false ? " · 历史版本" : ""}</span>
          </div>
        </div>
        <div className="match-source-card">
          <span className="match-source-icon resume" aria-hidden="true">CV</span>
          <div>
            <small>使用的简历</small>
            <strong>{match.resume_name ?? "简历名称未保存"}</strong>
            <span>{match.resume_version_number === null ? "版本未记录" : `简历 v${match.resume_version_number}`}</span>
          </div>
        </div>
      </div>
      {match.resume_available && match.resume_id ? (
        <a href={resumeDocumentUrl(match.resume_id, match.resume_version_id, { apiBaseUrl })} target="_blank" rel="noreferrer">
          查看参与匹配的简历原文件
        </a>
      ) : <p role="status">原简历已删除或不可访问，无法查看原文件；仍可查看匹配报告。</p>}
      {match.jd_available ? (
        <SavedJobCard
          apiBaseUrl={apiBaseUrl}
          resource={{
            kind: "saved_job",
            resourceId: match.jd_snapshot_id,
            title: `参与匹配的 JD v${match.jd_version ?? "未知"}`,
          }}
        />
      ) : <p role="status">原 JD 已删除或不可访问，无法查看原文；仍可查看匹配报告。</p>}
    </section>
  );
}
