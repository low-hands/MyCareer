import { MarkdownContent } from "./MarkdownContent";

const SECTION_LABELS: Record<string, { icon: string; title: string; tone: string }> = {
  "研究结论": { icon: "🔎", title: "关键发现", tone: "findings" },
  "面试前仍需确认": { icon: "💬", title: "面试前值得问清楚", tone: "questions" },
  "信息限制": { icon: "📌", title: "信息边界", tone: "limits" },
  "用户提供的检索线索": { icon: "📝", title: "调研线索", tone: "context" },
  "来源": { icon: "📚", title: "参考来源", tone: "sources" },
};
const CONFIDENCE: Record<string, string> = { high: "高", medium: "中", low: "低" };

export function CompanyReportContent({ body }: { body: string }) {
  // Markdown shortcut references link citations without rewriting existing links or code.
  const sourceReferences = Array.from(
    body.matchAll(/^### \[([^\]]+)\] \[[^\n]*?\]\((<https?:\/\/[^>\n]+>|https?:\/\/[^\s]+)\)/gm),
    (match) => `[${match[1]}]: ${match[2]}`,
  ).join("\n");
  const cite = (content: string) => `${content}\n\n${sourceReferences}`;
  const sections = body.replace(/^# 公司调研\s*\n/, "").split(/(?=^## )/m);
  return <div className="company-report" id="company-report-top">
    {sections.map((section, index) => {
      if (!section.trim()) return null;
      const heading = section.match(/^## (.+)\n?/);
      if (!heading) return <section className="company-report-overview" key={index}>
        <h3><span aria-hidden="true">🏢</span> 公司概览</h3>
        <MarkdownContent content={cite(section)} className="markdown-content" />
      </section>;
      const title = heading[1].trim();
      const meta = SECTION_LABELS[title];
      const content = section.slice(heading[0].length);
      if (!meta) return <MarkdownContent key={index} content={cite(section)} className="markdown-content" />;
      const parts = content.split(/(?=^### )/m).filter((part) => part.trim());
      return <section className={`company-report-section company-report-${meta.tone}`} key={index}>
        <h3><span aria-hidden="true">{meta.icon}</span> {meta.title}</h3>
        {title === "研究结论" ? <div className="company-findings">
          {parts.map((part, position) => {
            const kind = part.match(/^- 类型：(事实|推断|未知)\s*$/m);
            const confidence = part.match(/^- 置信度：(high|medium|low)\s*$/m);
            const cleaned = part.replace(/^- 类型：(事实|推断|未知)\s*$/m, "").replace(/^- 置信度：(high|medium|low)\s*$/m, "");
            return <article className="company-finding" key={position}>
              <div className="company-finding-meta"><span className="company-finding-number">{String(position + 1).padStart(2, "0")}</span>
                {kind ? <span className={`company-evidence ${kind[1] === "事实" ? "fact" : kind[1] === "推断" ? "inference" : "unknown"}`}>{kind[1] === "事实" ? "✓ 事实" : kind[1] === "推断" ? "↗ 推断" : "? 未知"}</span> : null}
                {confidence ? <span>置信度：{CONFIDENCE[confidence[1]]}</span> : null}
              </div>
              <MarkdownContent content={cite(cleaned)} className="markdown-content" />
            </article>;
          })}
        </div> : title === "来源" ? parts.map((part, position) => {
          const sourceHeading = part.match(/^### (.+)\n?/);
          if (!sourceHeading) return <MarkdownContent key={position} content={cite(part)} className="markdown-content" />;
          const excerpt = part.slice(sourceHeading[0].length);
          const sourceKey = sourceHeading[1].match(/^\[([^\]]+)\]/)?.[1] ?? `source-${position + 1}`;
          return <article className="company-source" id={`company-source-${sourceKey}`} key={position}>
            <MarkdownContent content={sourceHeading[1]} className="markdown-content" />
            {excerpt.trim() ? <details><summary>查看引用摘录</summary><MarkdownContent content={cite(excerpt)} className="markdown-content" /></details> : null}
            <a className="company-source-back" href="#company-report-top">↑ 返回正文</a>
          </article>;
        }) : <MarkdownContent content={cite(content)} className="markdown-content" />}
      </section>;
    })}
  </div>;
}
