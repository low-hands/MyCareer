import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CompanyReportContent } from "./CompanyReportContent";

describe("company report layout", () => {
  it("preserves findings, caveats, confidence and source links", () => {
    const html = renderToStaticMarkup(<CompanyReportContent body={'# 公司调研\n\n公司概况摘要\n\n> 报告已过期\n\n## 研究结论\n\n### 产品方向\n\n提供企业服务 [S1]\n\n- 类型：推断\n- 置信度：medium\n\n## 面试前仍需确认\n\n- 团队规模\n\n## 信息限制\n\n- 信息未核实\n\n## 来源\n\n### [S1] [企业官网](<https://example.com>) — 官方\n\n> 引用原文'} />);
    for (const text of ['公司概况摘要', '报告已过期', '产品方向', '提供企业服务 ', '推断', '置信度：中', '团队规模', '信息未核实', '引用原文', '官方']) expect(html).toContain(text);
    expect(html).toContain('href="https://example.com"');
    expect(html).toContain('href="https://example.com" target="_blank" rel="noreferrer noopener">S1</a>');
    expect(html).not.toContain('href="#company-source-S1"');
    expect(html).toContain('<details>');
    expect(html).not.toContain('<details open');
  });
  it("links citations throughout the report while preserving explicit links and code", () => {
    const html = renderToStaticMarkup(<CompanyReportContent body={'概览 [s1]\n\n## 历史补充\n\n补充 [S2] [missing] [S1](https://other.example) `[S1]`\n\n## 来源\n\n### [S1] [官网](<https://example.com/about_(team)>)\n\n### [S2] [新闻](https://news.example/article)'} />);
    expect(html).toContain('href="https://example.com/about_(team)" target="_blank" rel="noreferrer noopener">s1</a>');
    expect(html).toContain('href="https://news.example/article" target="_blank" rel="noreferrer noopener">S2</a>');
    expect(html).toContain('href="https://other.example" target="_blank" rel="noreferrer noopener">S1</a>');
    expect(html).toContain('<code>[S1]</code>');
    expect(html).toContain('[missing]');
  });
  it("keeps unfamiliar historical sections and safe Markdown rendering", () => {
    const html = renderToStaticMarkup(<CompanyReportContent body={'## 历史补充\n\n保留内容\n\n![图片](https://example.com/track.png)\n\n<script>alert(1)</script>'} />);
    expect(html).toContain('历史补充');
    expect(html).toContain('保留内容');
    expect(html).not.toContain('<img');
    expect(html).not.toContain('<script');
  });
});
