import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

/** Render agent-authored Markdown without enabling raw HTML or remote images. */
export function MarkdownContent({ content, className }: MarkdownContentProps) {
  return (
    <div className={className}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ children, href, ...props }) => {
            if (!href || !/^https?:\/\//i.test(href)) {
              return <span>{children}</span>;
            }
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
