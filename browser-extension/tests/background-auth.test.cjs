const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function background(fetchImpl) {
  const source = fs.readFileSync(
    path.join(__dirname, "..", "background.js"),
    "utf8",
  );
  const context = {
    URL,
    fetch: fetchImpl,
    chrome: {
      runtime: { onMessage: { addListener() {} } },
      storage: {
        local: {
          async get() {
            return { careerAgentCaptureApiKey: "cak_capture" };
          },
          async set() {},
        },
      },
    },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return context;
}

test("capture transport authenticates and never submits a user identity", async () => {
  const requests = [];
  const runtime = background(async (url, request) => {
    requests.push([url, request]);
    return new Response(JSON.stringify({ job_posting_id: "job-1" }), {
      status: 200,
    });
  });

  await runtime.saveJob(
    {
      source_url: "https://www.zhipin.com/job_detail/abc.html",
      title: "AI Engineer",
      company_name: "Acme",
      description: "Build reliable agents",
    },
    { tab: { url: "https://www.zhipin.com/job_detail/abc.html" } },
  );

  const request = requests[0][1];
  assert.equal(request.headers.Authorization, "Bearer cak_capture");
  assert.deepEqual(JSON.parse(request.body), {
    source_url: "https://www.zhipin.com/job_detail/abc.html",
    title: "AI Engineer",
    company_name: "Acme",
    description: "Build reliable agents",
    city: null,
    salary: null,
    experience: null,
    education: null,
  });
});

test("closure transport uses the same scoped credential and URL-only body", async () => {
  const requests = [];
  const runtime = background(async (url, request) => {
    requests.push([url, request]);
    return new Response(JSON.stringify({ matched: true }), { status: 200 });
  });

  await runtime.reportClosed(
    "https://www.zhipin.com/job_detail/abc.html",
    { tab: { url: "https://www.zhipin.com/job_detail/abc.html" } },
  );

  const request = requests[0][1];
  assert.equal(request.headers.Authorization, "Bearer cak_capture");
  assert.deepEqual(JSON.parse(request.body), {
    source_url: "https://www.zhipin.com/job_detail/abc.html",
  });
});
