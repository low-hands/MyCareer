import { useState } from "react";

import type {
  JobAnalysisResult,
  JobSeniority,
  QuotedFinding,
  RequirementKind,
  RequirementTier,
  TieredRequirement,
} from "../api/client";

export const SENIORITY_LABELS: Record<JobSeniority, string> = {
  fresh_graduate: "应届",
  junior: "初级",
  mid: "中级",
  senior: "高级",
  lead: "管理",
};

export const TIER_ORDER: readonly RequirementTier[] = ["S", "A", "B", "C"];

export const TIER_LABELS: Record<RequirementTier, string> = {
  S: "S · 硬性门槛",
  A: "A · 核心要求",
  B: "B · 加分项",
  C: "C · 可选",
};

const KIND_LABELS: Record<RequirementKind, string> = {
  fact: "事实",
  inference: "推断",
};

/** Requirements grouped by tier, S first, keeping the analysis' own order inside a tier. */
export function requirementsByTier(
  requirements: readonly TieredRequirement[],
): { tier: RequirementTier; items: TieredRequirement[] }[] {
  return TIER_ORDER.map((tier) => ({
    tier,
    items: requirements.filter((item) => item.tier === tier),
  })).filter((group) => group.items.length > 0);
}

function QuotedList({ items }: { items: QuotedFinding[] }) {
  return (
    <ul className="jd-quoted-list">
      {items.map((item) => (
        <li key={`${item.text}|${item.jd_quote}`}>
          <span>{item.text}</span>
          <blockquote>{item.jd_quote}</blockquote>
        </li>
      ))}
    </ul>
  );
}

function PlainList({ items }: { items: string[] }) {
  return (
    <ul>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

interface JobAnalysisDetailsProps {
  analysis: JobAnalysisResult;
  /** The JD version the result was computed on, when known. */
  version: number | null;
  /** True when the posting has a newer JD snapshot than `version`. */
  stale: boolean;
}

/**
 * The JD-only analysis of one saved job, collapsed to its headline by default.
 *
 * Every requirement is shown with the JD line it rests on and whether it is
 * stated or inferred, so the reader can check the reading against the text
 * instead of trusting a label. Nothing here is about a resume: fit lives in
 * the separate match result.
 */
export function JobAnalysisDetails({ analysis, version, stale }: JobAnalysisDetailsProps) {
  const [expanded, setExpanded] = useState(false);
  const groups = requirementsByTier(analysis.requirements);
  return (
    <div className="jd-analysis">
      <div className="jd-analysis-headline">
        <span className="jd-seniority">{SENIORITY_LABELS[analysis.seniority]}</span>
        <span className="jd-objective">{analysis.core_objective}</span>
        {version !== null ? (
          <span className={`jd-analysis-version${stale ? " stale" : ""}`}>
            基于 JD v{version}
            {stale ? " · 当前版本待分析" : ""}
          </span>
        ) : null}
      </div>
      {analysis.core_competencies.length > 0 ? (
        <div className="analysis-section">
          <strong>核心能力</strong>
          <div className="skill-chips">
            {analysis.core_competencies.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
      ) : null}
      <button
        type="button"
        className="jd-requirements-toggle"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? "收起分层要求" : `展开分层要求（${analysis.requirements.length} 条）`}
        <svg aria-hidden="true" width="12" height="12" viewBox="0 0 12 12" className={expanded ? "is-open" : undefined}>
          <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {expanded ? (
        <div className="jd-analysis-body">
          {groups.map((group) => (
            <div className="analysis-section jd-tier" key={group.tier}>
              <strong>{TIER_LABELS[group.tier]}</strong>
              <ul className="jd-requirements">
                {group.items.map((item) => (
                  <li key={`${item.text}|${item.jd_quote}`}>
                    <span className={`jd-kind jd-kind-${item.kind}`}>{KIND_LABELS[item.kind]}</span>
                    <span>{item.text}</span>
                    <blockquote>{item.jd_quote}</blockquote>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {analysis.implicit_requirements.length > 0 ? (
            <div className="analysis-section">
              <strong>隐性要求</strong>
              <QuotedList items={analysis.implicit_requirements} />
            </div>
          ) : null}
          {analysis.ats_keywords.length > 0 ? (
            <div className="analysis-section">
              <strong>ATS 关键词</strong>
              <div className="skill-chips">
                {analysis.ats_keywords.map((item) => (
                  <span key={item}>{item}</span>
                ))}
              </div>
            </div>
          ) : null}
          {analysis.hr_focus.length > 0 ? (
            <div className="analysis-section"><strong>HR 关注点</strong><PlainList items={analysis.hr_focus} /></div>
          ) : null}
          {analysis.hiring_manager_focus.length > 0 ? (
            <div className="analysis-section"><strong>Hiring Manager 关注点</strong><PlainList items={analysis.hiring_manager_focus} /></div>
          ) : null}
          {analysis.likely_interview_topics.length > 0 ? (
            <div className="analysis-section"><strong>可能的面试话题</strong><PlainList items={analysis.likely_interview_topics} /></div>
          ) : null}
          {analysis.red_flags.length > 0 ? (
            <div className="analysis-section jd-red-flags">
              <strong>红旗信号</strong>
              <QuotedList items={analysis.red_flags} />
            </div>
          ) : null}
          {analysis.information_gaps.length > 0 ? (
            <div className="analysis-section"><strong>待向 HR 确认</strong><PlainList items={analysis.information_gaps} /></div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
