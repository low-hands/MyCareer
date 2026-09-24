import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Children, type ReactNode } from "react";

interface MarkdownContentProps {
  content: string;
  className?: string;
  highlightMatchStatus?: boolean;
}

const MATCH_STATUS: Record<string, { tone: string; icon: string }> = {
  "匹配": { tone: "positive", icon: "✅" },
  "强匹配": { tone: "positive", icon: "✅" },
  "部分匹配": { tone: "partial", icon: "🟡" },
  "中等匹配": { tone: "partial", icon: "🟡" },
  "未体现": { tone: "missing", icon: "🧩" },
  "弱匹配": { tone: "missing", icon: "🧩" },
  "不明确": { tone: "unknown", icon: "🔎" },
  "证据不足": { tone: "unknown", icon: "🔎" },
};

function statusBadge(label: string) {
  const status = MATCH_STATUS[label];
  return <span className={`match-status-badge match-status-${status.tone}`}><span aria-hidden="true">{status.icon}</span>{label}</span>;
}

function requirementStatus(children: ReactNode) {
  return Children.map(children, (child) => {
    if (typeof child !== "string") return child;
    const match = child.match(/^(\d+\.\s*)?\[(匹配|部分匹配|未体现|不明确)\]\s*/);
    return match ? <>{match[1]}{statusBadge(match[2])}{child.slice(match[0].length)}</> : child;
  });
}

/** Render agent-authored Markdown without enabling raw HTML or remote images. */
export function MarkdownContent({ content, className, highlightMatchStatus = false }: MarkdownContentProps) {
  return (
    <div className={className}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          h2: ({ children }) => {
            const icons: Record<string, string> = { "建议": "💡", "需要补充的信息": "🙋", "分析限制": "📌", "要求逐项核对": "🧭" };
            const icon = highlightMatchStatus && typeof children === "string" ? icons[children] : undefined;
            return <h2>{icon ? <span aria-hidden="true">{icon} </span> : null}{children}</h2>;
          },
          h3: ({ children }) => <h3>{highlightMatchStatus ? requirementStatus(children) : children}</h3>,
          p: ({ children }) => {
            const text = typeof children === "string" ? children : null;
            const match = highlightMatchStatus && text?.match(/^整体判断[：:]\s*(强匹配|中等匹配|弱匹配|证据不足)\s*$/);
            return match ? <p className="match-overall-verdict">整体判断 {statusBadge(match[1])}</p> : <p>{children}</p>;
          },
          a: ({ children, href, ...props }) => {
            if (!href || !(/^(https?:\/\/|#)/i.test(href))) {
              return <span>{children}</span>;
            }
            if (href.startsWith("#")) return <a {...props} href={href}>{children}</a>;
            return (
              <a {...props} href={href} target="_blank" rel="noreferrer noopener">
                {children}
              </a>
            );
          },
          img: ({ alt }) => <span>{alt ? `[图片：${alt}]` : "[图片]"}</span>,
        }}
      >
        {content}
      </Markdown>
    </div>
  );
}
