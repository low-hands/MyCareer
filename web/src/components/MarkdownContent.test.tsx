import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { MarkdownContent } from "./MarkdownContent";

describe("MarkdownContent", () => {
  it("highlights match verdicts and requirement statuses without changing evidence text", () => {
    const content = "整体判断：弱匹配\n\n### 1. [部分匹配] 开发服务\n\n### 2. [未体现] Python\n\n### 3. [匹配] 协作\n\n### 4. [不明确] 学历\n\n> 原文含有 [部分匹配] 字样";
    const html = renderToStaticMarkup(<MarkdownContent content={content} highlightMatchStatus />);
    expect(html).toContain("match-overall-verdict");
    for (const tone of ["positive", "partial", "missing", "unknown"]) {
      expect(html).toContain(`match-status-${tone}`);
    }
    expect(html).toContain("原文含有 [部分匹配] 字样");
    expect(renderToStaticMarkup(<MarkdownContent content={content} />)).not.toContain("match-status-badge");
  });

  it("renders presenter Markdown instead of exposing syntax markers", () => {
    const html = renderToStaticMarkup(
      <MarkdownContent
        content={"# 公司调研\n\n> 时效提示\n\n- 结论\n\n[来源](https://example.com)"}
      />,
    );

    expect(html).toContain("<h1>公司调研</h1>");
    expect(html).toContain("<blockquote>");
    expect(html).toContain("<li>结论</li>");
    expect(html).toContain('rel="noreferrer noopener"');
    expect(html).not.toContain("# 公司调研");
  });

  it("does not execute raw HTML or request remote Markdown images", () => {
    const html = renderToStaticMarkup(
      <MarkdownContent
        content={'<script>alert(1)</script>\n\n![tracking](https://example.com/pixel.png)'}
      />,
    );

    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<img");
    expect(html).not.toContain("pixel.png");
    expect(html).toContain("[图片：tracking]");
  });

  it("only makes http and https links clickable", () => {
    const html = renderToStaticMarkup(
      <MarkdownContent
        content={"[unsafe](javascript:alert(1)) [mail](mailto:a@example.com) [safe](https://example.com)"}
      />,
    );

    expect(html).not.toContain("javascript:");
    expect(html).not.toContain("mailto:");
    expect(html).toContain('href="https://example.com"');
  });
});
