const assert = require("node:assert/strict");
const test = require("node:test");

const parser = require("../boss-parser.js");

function node(text) {
  return { innerText: text, textContent: text };
}

function documentFixture(fields, bodyText = "") {
  function nodesFor(selector) {
    const value = fields[selector];
    if (!value) return [];
    if (Array.isArray(value)) return value;
    if (typeof value === "object") return [value];
    return [node(value)];
  }

  return {
    body: node(bodyText),
    querySelector(selector) {
      return nodesFor(selector)[0] || null;
    },
    querySelectorAll(selector) {
      return nodesFor(selector);
    },
  };
}

test("extracts only structured current-detail fields", () => {
  const documentRef = documentFixture({
    ".job-title": "AI 产品经理",
    ".company-name": "示例科技",
    ".job-banner": "上海 3-5年 本科 25-35K",
    ".job-description": "职位描述\n负责 AI 产品规划、需求分析和交付。\n任职要求\n熟悉大模型产品。",
  });

  const result = parser.extract(documentRef, {
    href: "https://www.zhipin.com/job_detail/abc.html?securityId=secret",
  });

  assert.equal(result.status, "ready");
  assert.equal(result.job.title, "AI 产品经理");
  assert.equal(result.job.company_name, "示例科技");
  assert.equal(result.job.source_url, "https://www.zhipin.com/job_detail/abc.html");
  assert.equal(result.job.salary, "25-35K");
  assert.equal(result.job.city, "上海");
  assert.equal(result.job.experience, "3-5年");
  assert.equal(result.job.education, "本科");
  assert.doesNotMatch(JSON.stringify(result), /secret/);
});

test("reports security verification instead of extracting", () => {
  const result = parser.extract(
    documentFixture({}, "请完成安全验证 请滑动"),
    { href: "https://www.zhipin.com/web/geek/job" },
  );

  assert.deepEqual(result, { status: "security_check" });
});

test("does not treat a search page as a complete job", () => {
  const result = parser.extract(
    documentFixture({ ".job-title": "AI 产品经理" }),
    { href: "https://www.zhipin.com/web/geek/job" },
  );

  assert.equal(result.status, "not_ready");
  assert.deepEqual(result.missing_fields, ["company_name", "description"]);
});

test("ignores a hidden company left behind by client-side navigation", () => {
  const staleCompany = {
    ...node("拼多多集团-PDD"),
    hidden: true,
    getClientRects: () => [],
  };
  const currentCompany = {
    ...node("字节跳动"),
    hidden: false,
    getClientRects: () => [{ width: 120, height: 24 }],
  };
  const documentRef = documentFixture({
    ".job-title": "27届校招-后端开发工程师-番茄/红果短剧",
    ".sider-company .company-name": [staleCompany, currentCompany],
    ".job-description": "职位描述\n负责字节跳动番茄小说服务端开发工作。",
  });

  const result = parser.extract(documentRef, {
    href: "https://www.zhipin.com/job_detail/bytedance.html",
  });

  assert.equal(result.status, "ready");
  assert.equal(result.job.company_name, "字节跳动");
});

test("prefers the labelled job description over a longer company section", () => {
  function sectionNode(text, heading) {
    const section = {
      innerText: `${heading}\n${text}`,
      textContent: `${heading}\n${text}`,
      querySelector: () => node(heading),
    };
    return {
      ...node(text),
      closest: (selector) => selector.includes("[hidden]") ? null : section,
    };
  }
  const jd = "负责多模态大模型算法研发。\n任职要求：熟悉 Python 与深度学习。";
  const company = "智谱华章是一家专注人工智能技术的公司。".repeat(20);
  const documentRef = documentFixture({
    ".job-title": "大模型算法实习生",
    ".company-name": "智谱华章",
    ".job-detail-section .text": [
      sectionNode(jd, "职位描述"),
      sectionNode(company, "公司基本信息"),
    ],
  });

  const result = parser.extract(documentRef, {
    href: "https://www.zhipin.com/job_detail/zhipu.html",
  });

  assert.equal(result.status, "ready");
  assert.equal(result.job.description, jd);
  assert.doesNotMatch(result.job.description, /专注人工智能技术的公司/);
});

test("rejects non-BOSS URLs", () => {
  assert.equal(parser.canonicalPageUrl("https://example.com/job_detail/abc.html"), "");
});

test("a closed posting is reported as such instead of being captured", () => {
  // The one fact a closed page still carries. Nothing else in the system can
  // observe it: postings are re-read only when the user opens them, never
  // polled.
  const documentRef = documentFixture(
    { ".job-title": "AI 产品经理" },
    "该职位已关闭，看看其他机会",
  );

  const result = parser.extract(documentRef, {
    href: "https://www.zhipin.com/job_detail/abc.html",
  });

  assert.equal(result.status, "closed");
  assert.equal(result.job, undefined);
});

test("a page we were not allowed to see is never read as a closure", () => {
  // Being blocked says nothing about whether the job is still open. Recording
  // "closed" from a login wall or a captcha would turn our own lack of access
  // into an employer decision.
  for (const [text, expected] of [
    ["请登录后查看该职位", "login_required"],
    ["请完成安全验证后继续", "security_check"],
  ]) {
    const documentRef = documentFixture({ ".job-title": "AI 产品经理" }, text);
    assert.equal(parser.pageBarrier(documentRef), expected);
  }
});

test("closure wording variants are all recognised", () => {
  for (const text of [
    "职位已关闭",
    "该职位已下线",
    "职位已下架",
    "该职位已失效",
    "职位不存在",
  ]) {
    assert.equal(parser.closedPosting(documentFixture({}, text)), true, text);
  }
  assert.equal(
    parser.closedPosting(documentFixture({}, "职位描述：负责 AI 产品规划")),
    false,
  );
});

test("takes the job name from the banner heading, not the block that also holds the salary", () => {
  const documentRef = documentFixture({
    ".job-banner .name h1": "Agent开发实习",
    ".job-banner .name": "Agent开发实习 200-250元/天",
    ".job-banner .salary": "200-250元/天",
    ".company-name": "量霸科技",
    ".job-description": "职位描述\n负责 Agent 应用的设计、开发和评测。\n任职要求\n熟悉 Python 和大模型应用开发。",
  });

  const result = parser.extract(documentRef, { href: "https://www.zhipin.com/job_detail/abc.html" });

  assert.equal(result.job.title, "Agent开发实习");
  assert.equal(result.job.salary, "200-250元/天");
});
