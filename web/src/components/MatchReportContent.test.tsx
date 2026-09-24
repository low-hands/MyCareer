import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MatchReportContent } from "./MatchReportContent";

describe("match report layout", () => {
  it("keeps conclusions visible and preserves original evidence in expandable sections", () => {
    const html = renderToStaticMarkup(<MatchReportContent hasVerdict body={'# 简历与岗位匹配\n\n整体判断：弱匹配\n\n总体原因\n\n## 要求逐项核对\n\n### 1. [部分匹配] Python\n\n> JD：开发服务\n\n判断原因\n\n简历证据：\n- 第一页：脚本开发\n\n## 建议\n\n- 补充项目'} />);
    expect(html).toContain('class="match-requirement"');
    expect(html).toContain('<details');
    for (const text of ['总体原因', '判断原因', '开发服务', '第一页：脚本开发', '补充项目']) expect(html).toContain(text);
    expect(html).not.toContain('整体判断：');
    expect(html).not.toContain('<details open');
  });
  it("preserves unfamiliar historical reports", () => {
    const html = renderToStaticMarkup(<MatchReportContent hasVerdict={false} body={'# 历史报告\n\n整体判断：弱匹配\n\n自由格式内容'} />);
    expect(html).toContain('历史报告');
    expect(html).toContain('自由格式内容');
    expect(html).toContain('弱匹配');
  });
});
