import { MarkdownContent } from "./MarkdownContent";

/** Split only the presenter's known headings; unfamiliar historical content stays intact. */
export function MatchReportContent({ body, hasVerdict }: { body: string; hasVerdict: boolean }) {
  const content = hasVerdict ? body
    .replace(/^# 简历与岗位匹配\s*\n/, "")
    .replace(/^整体判断[：:]\s*(强匹配|中等匹配|弱匹配|证据不足)\s*$/m, "") : body;
  const sections = content.split(/(?=^## )/m);
  return <div className="match-report-content">
    {sections.map((section, index) => {
      if (!section.startsWith("## 要求逐项核对")) {
        return <MarkdownContent key={index} content={section} className={`markdown-content match-report-section ${section.startsWith("## 建议") ? "match-report-advice" : section.startsWith("## 需要补充") ? "match-report-questions" : ""}`} highlightMatchStatus />;
      }
      const [heading, ...requirements] = section.split(/(?=^### \d+\. \[(?:匹配|部分匹配|未体现|不明确)\])/m);
      return <section className="match-requirements" key={index}>
        <MarkdownContent content={heading.replace(/^## 要求逐项核对/, "## 🧭 要求逐项核对")} className="markdown-content match-report-section" />
        {requirements.map((requirement, position) => {
          const lines = requirement.split("\n");
          const title = lines.shift()!;
          const evidenceIndex = lines.findIndex((line) => line.trim() === "简历证据：");
          const main = evidenceIndex < 0 ? lines : lines.slice(0, evidenceIndex);
          const quotes: string[] = [];
          const rationale: string[] = [];
          for (const line of main) (line.startsWith(">") ? quotes : rationale).push(line);
          const evidence = evidenceIndex < 0 ? "" : lines.slice(evidenceIndex + 1).join("\n");
          return <article className="match-requirement" key={position}>
            <MarkdownContent content={title} className="markdown-content match-requirement-title" highlightMatchStatus />
            <MarkdownContent content={rationale.join("\n")} className="markdown-content match-requirement-rationale" />
            {quotes.length || evidence ? <details className="match-evidence">
              <summary>📎 查看岗位原文与简历证据</summary>
              {quotes.length ? <MarkdownContent content={quotes.join("\n")} className="markdown-content" /> : null}
              {evidence ? <><h4>📝 简历证据</h4><MarkdownContent content={evidence} className="markdown-content" /></> : null}
            </details> : null}
          </article>;
        })}
      </section>;
    })}
  </div>;
}
