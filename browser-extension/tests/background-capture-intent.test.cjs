const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const INTENT = `capint_${"a".repeat(32)}`;
const BOSS_SEARCH = "https://www.zhipin.com/web/geek/job?query=AI&city=101020100";
const BOSS_DETAIL = "https://www.zhipin.com/job_detail/abc.html";
const APP_TAB = { tab: { id: 1, url: "http://127.0.0.1:5173/" } };

function background(fetchImpl, { now = () => 1_000_000, persistedLocal } = {}) {
  const source = fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8");
  const local = persistedLocal || { careerAgentCaptureApiKey: "cak_capture" };
  const created = [];
  const sent = [];
  let nextTabId = 100;
  const context = {
    URL,
    fetch: fetchImpl,
    Date: Object.assign(function DateShim(...args) {
      return new Date(...args);
    }, { now, parse: Date.parse }),
    chrome: {
      runtime: {
        onMessage: { addListener() {} },
        onStartup: { addListener() {} },
      },
      tabs: {
        onCreated: { addListener() {} },
        onRemoved: { addListener() {} },
        async create({ url }) {
          const tab = { id: nextTabId++, url };
          created.push(tab);
          return tab;
        },
        async query() {
          return [{ id: 1 }, { id: 2 }];
        },
        async sendMessage(tabId, message) {
          sent.push([tabId, message]);
        },
      },
      storage: {
        local: {
          async get(key) {
            return { [key]: local[key] };
          },
          async set(values) {
            Object.assign(local, values);
          },
          async remove(key) {
            delete local[key];
          },
        },
      },
    },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  // Keep the old test-facing name: callers care about the stored bindings,
  // not which Chrome storage area owns them.
  return { runtime: context, session: local, local, created, sent };
}

function saveResponse(extra = {}) {
  return async () =>
    new Response(
      JSON.stringify({ job_posting_id: "job-1", jd_snapshot_id: "snap-1", ...extra }),
      { status: 200 },
    );
}

test("opening a search binds the intent to the new tab and keeps it out of the URL", async () => {
  const { runtime, session, created } = background(saveResponse());

  const result = await runtime.openJobSearch(
    {
      url: BOSS_SEARCH,
      capture_intent_id: INTENT,
      capture_intent_expires_at: "2026-09-12T14:00:00+00:00",
    },
    APP_TAB,
  );

  assert.deepEqual(JSON.parse(JSON.stringify(result)), { tab_id: 100, bound: true });
  assert.equal(created[0].url, BOSS_SEARCH);
  assert.ok(!created[0].url.includes(INTENT));
  assert.equal(session.careerAgentTabIntents["100"].intent_id, INTENT);
  assert.equal(
    session.careerAgentTabIntents["100"].expires_at,
    Date.parse("2026-09-12T14:00:00+00:00"),
  );
});

test("an extension reload keeps the intent for an already-open BOSS tab", async () => {
  const requests = [];
  const persistedLocal = { careerAgentCaptureApiKey: "cak_capture" };
  const first = background(saveResponse(), { persistedLocal });
  await first.runtime.openJobSearch(
    { url: BOSS_SEARCH, capture_intent_id: INTENT },
    APP_TAB,
  );

  // A new background context models Chrome reloading the unpacked extension.
  const reloaded = background(async (url, request) => {
    requests.push(JSON.parse(request.body));
    return saveResponse()();
  }, { persistedLocal });
  await reloaded.runtime.saveJob(
    { source_url: BOSS_DETAIL, title: "AI Engineer", company_name: "质谱华章", description: "Build" },
    { tab: { id: 100, url: BOSS_DETAIL } },
  );

  assert.equal(requests[0].capture_intent_id, INTENT);
});

test("only the Career Agent page may open, and only a BOSS URL", async () => {
  const { runtime, created } = background(saveResponse());

  await assert.rejects(
    runtime.openJobSearch({ url: BOSS_SEARCH, capture_intent_id: INTENT }, {
      tab: { id: 9, url: "https://evil.example/" },
    }),
    (error) => error.code === "UNTRUSTED_PAGE",
  );
  await assert.rejects(
    runtime.openJobSearch({ url: "https://evil.example/?q=1", capture_intent_id: INTENT }, APP_TAB),
    (error) => error.code === "UNTRUSTED_URL",
  );
  await assert.rejects(
    runtime.openJobSearch({ url: BOSS_SEARCH, capture_intent_id: "conv-1" }, APP_TAB),
    (error) => error.code === "INVALID_INTENT",
  );
  assert.equal(created.length, 0);
});

test("a save from a bound tab carries its intent; other tabs carry none", async () => {
  const requests = [];
  const { runtime, sent } = background(async (url, request) => {
    requests.push(JSON.parse(request.body));
    return saveResponse({
      conversation_id: "conv-1",
      capture_event_id: `jobcap_${"b".repeat(32)}`,
      capture_event_created: true,
    })();
  });
  await runtime.openJobSearch({ url: BOSS_SEARCH, capture_intent_id: INTENT }, APP_TAB);
  const job = {
    source_url: BOSS_DETAIL,
    title: "AI Engineer",
    company_name: "Acme",
    description: "Build reliable agents",
  };

  await runtime.saveJob(job, { tab: { id: 100, url: BOSS_DETAIL } });
  await runtime.saveJob(job, { tab: { id: 555, url: BOSS_DETAIL } });

  assert.equal(requests[0].capture_intent_id, INTENT);
  assert.equal("capture_intent_id" in requests[1], false);
  // Every open Career Agent page is nudged with the ids the backend returned.
  assert.deepEqual(
    sent.map(([tabId, message]) => [tabId, message.type, message.conversation_id]),
    [
      [1, "CAREER_AGENT_JOB_CAPTURED", "conv-1"],
      [2, "CAREER_AGENT_JOB_CAPTURED", "conv-1"],
      [1, "CAREER_AGENT_JOB_CAPTURED", "conv-1"],
      [2, "CAREER_AGENT_JOB_CAPTURED", "conv-1"],
    ],
  );
});

test("a duplicate save does not nudge the page twice", async () => {
  let calls = 0;
  const { runtime, sent } = background(async () => {
    calls += 1;
    return saveResponse({
      conversation_id: "conv-1",
      capture_event_id: `jobcap_${"b".repeat(32)}`,
      capture_event_created: calls === 1,
    })();
  });
  await runtime.openJobSearch({ url: BOSS_SEARCH, capture_intent_id: INTENT }, APP_TAB);
  const job = { source_url: BOSS_DETAIL, title: "A", company_name: "B", description: "C" };

  await runtime.saveJob(job, { tab: { id: 100, url: BOSS_DETAIL } });
  await runtime.saveJob(job, { tab: { id: 100, url: BOSS_DETAIL } });

  assert.equal(sent.length, 2);
});

test("a detail tab opened from the search tab inherits the intent; closing forgets it", async () => {
  const requests = [];
  const { runtime, session } = background(async (url, request) => {
    requests.push(JSON.parse(request.body));
    return saveResponse()();
  });
  await runtime.openJobSearch({ url: BOSS_SEARCH, capture_intent_id: INTENT }, APP_TAB);

  await runtime.inheritTabIntent(100, 101);
  await runtime.inheritTabIntent(999, 102);
  await runtime.saveJob(
    { source_url: BOSS_DETAIL, title: "A", company_name: "B", description: "C" },
    { tab: { id: 101, url: BOSS_DETAIL } },
  );
  await runtime.forgetTabIntent(101);
  await runtime.forgetTabIntent(100);

  assert.equal(requests[0].capture_intent_id, INTENT);
  assert.equal("102" in session.careerAgentTabIntents, false);
  assert.deepEqual(Object.keys(session.careerAgentTabIntents), []);
});

test("an expired binding is dropped instead of sent", async () => {
  const requests = [];
  let clock = 1_000_000;
  const { runtime, session } = background(
    async (url, request) => {
      requests.push(JSON.parse(request.body));
      return saveResponse()();
    },
    { now: () => clock },
  );
  await runtime.openJobSearch(
    { url: BOSS_SEARCH, capture_intent_id: INTENT, capture_intent_expires_at: "1970-01-01T00:20:00Z" },
    APP_TAB,
  );
  clock = 20 * 60 * 1000 + 1;

  await runtime.saveJob(
    { source_url: BOSS_DETAIL, title: "A", company_name: "B", description: "C" },
    { tab: { id: 100, url: BOSS_DETAIL } },
  );

  assert.equal("capture_intent_id" in requests[0], false);
  assert.deepEqual(Object.keys(session.careerAgentTabIntents), []);
});

test("concurrent bindings for different tabs are both kept", async () => {
  const { runtime, session } = background(saveResponse());
  const second = `capint_${"c".repeat(32)}`;

  await Promise.all([
    runtime.bindTabIntent(100, { intent_id: INTENT, expires_at: 5_000_000 }),
    runtime.bindTabIntent(200, { intent_id: second, expires_at: 5_000_000 }),
  ]);

  assert.equal(session.careerAgentTabIntents["100"].intent_id, INTENT);
  assert.equal(session.careerAgentTabIntents["200"].intent_id, second);
});

test("bind, inherit and forget interleaved concurrently converge on the right bindings", async () => {
  const { runtime, session } = background(saveResponse());
  const second = `capint_${"c".repeat(32)}`;
  await runtime.bindTabIntent(100, { intent_id: INTENT, expires_at: 5_000_000 });

  await Promise.all([
    runtime.inheritTabIntent(100, 101),
    runtime.bindTabIntent(200, { intent_id: second, expires_at: 5_000_000 }),
    runtime.forgetTabIntent(100),
    runtime.inheritTabIntent(200, 201),
  ]);

  assert.deepEqual(Object.keys(session.careerAgentTabIntents).sort(), ["101", "200", "201"]);
  assert.equal(session.careerAgentTabIntents["101"].intent_id, INTENT);
  assert.equal(session.careerAgentTabIntents["201"].intent_id, second);
});
