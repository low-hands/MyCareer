const assert = require("node:assert/strict");
const test = require("node:test");

const parser = require("../boss-parser.js");

function node(text) {
  return { innerText: text, textContent: text };
}

function documentFixture(fields, bodyText = "") {
  return {
    body: node(bodyText),
    querySelector(selector) {
      const value = fields[selector];
      return value ? node(value) : null;
    },
    querySelectorAll(selector) {
      const value = fields[selector];
      return value ? [node(value)] : [];
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

test("rejects non-BOSS URLs", () => {
  assert.equal(parser.canonicalPageUrl("https://example.com/job_detail/abc.html"), "");
});
