const API_HOSTS = ["http://127.0.0.1:8000", "http://localhost:8000"];
const API_ENDPOINTS = API_HOSTS.map((host) => `${host}/v1/browser-captures/jobs`);
const CLOSURE_ENDPOINTS = API_HOSTS.map(
  (host) => `${host}/v1/browser-captures/job-closures`,
);
const APP_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"];
const APP_TAB_PATTERNS = APP_ORIGINS.map((origin) => `${origin}/*`);
const INTENT_PATTERN = /^capint_[a-f0-9]{32}$/;
// Intents are bound to BOSS *tabs*, never written into a BOSS URL. Session
// storage outlives a service-worker restart but not the browser, which is
// also the lifetime of the tabs it describes.
const TAB_INTENTS_KEY = "careerAgentTabIntents";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "CAREER_AGENT_OPEN_JOB_SEARCH") {
    openJobSearch(message, sender)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({
        ok: false,
        code: error.code || "OPEN_FAILED",
        message: error.message || "无法打开搜索页",
      }));
    return true;
  }

  if (message?.type === "CAREER_AGENT_SET_CAPTURE_CREDENTIAL") {
    const apiKey = String(message.api_key || "").trim();
    if (!apiKey.startsWith("cak_")) {
      sendResponse({ ok: false, code: "INVALID_CAPTURE_CREDENTIAL" });
      return false;
    }
    chrome.storage.local.set({
      careerAgentCaptureApiKey: apiKey,
    }).then(() => {
      sendResponse({ ok: true });
    });
    return true;
  }

  if (message?.type === "CAREER_AGENT_REPORT_CLOSED") {
    reportClosed(message.source_url, sender)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({
        ok: false,
        code: error.code || "CLOSURE_FAILED",
        message: error.message || "标记失败",
      }));
    return true;
  }
  if (message?.type !== "CAREER_AGENT_SAVE_JOB") return false;
  saveJob(message.job, sender)
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({
      ok: false,
      code: error.code || "CAPTURE_FAILED",
      message: error.message || "岗位保存失败",
    }));
  return true;
});

// BOSS opens details in new tabs; a tab spawned from a bound tab inherits its
// intent so the save made two clicks later still knows which search it came
// from. Removing a tab drops its binding.
if (typeof chrome !== "undefined" && chrome.tabs?.onCreated) {
  chrome.tabs.onCreated.addListener((tab) => {
    if (tab.openerTabId === undefined || tab.id === undefined) return;
    inheritTabIntent(tab.openerTabId, tab.id).catch(() => undefined);
  });
  chrome.tabs.onRemoved.addListener((tabId) => {
    forgetTabIntent(tabId).catch(() => undefined);
  });
}

/** Open the agent's BOSS search in a fresh tab bound to its capture intent.
 *
 * Only the Career Agent page may ask, and only for a BOSS URL. The intent id
 * is kept beside the tab id here and is never appended to the URL BOSS sees.
 */
async function openJobSearch(message, sender) {
  const senderUrl = sender.tab?.url || sender.url || "";
  if (!isAppUrl(senderUrl)) {
    throw captureError("UNTRUSTED_PAGE", "只能从 Career Agent 页面打开搜索");
  }
  const url = String(message.url || "");
  if (!isBossUrl(url)) {
    throw captureError("UNTRUSTED_URL", "只能打开 BOSS 直聘的搜索页");
  }
  const intentId = message.capture_intent_id ? String(message.capture_intent_id) : null;
  if (intentId !== null && !INTENT_PATTERN.test(intentId)) {
    throw captureError("INVALID_INTENT", "采集任务标识无效");
  }
  const tab = await chrome.tabs.create({ url, active: true });
  if (intentId !== null && tab?.id !== undefined) {
    const expiresAt = Date.parse(String(message.capture_intent_expires_at || ""));
    await bindTabIntent(tab.id, {
      intent_id: intentId,
      expires_at: Number.isFinite(expiresAt) ? expiresAt : Date.now() + 2 * 60 * 60 * 1000,
    });
  }
  return { tab_id: tab?.id ?? null, bound: intentId !== null };
}

async function readTabIntents() {
  if (!chrome.storage?.session) return {};
  const stored = await chrome.storage.session.get(TAB_INTENTS_KEY);
  const intents = stored?.[TAB_INTENTS_KEY];
  return intents && typeof intents === "object" ? { ...intents } : {};
}

async function bindTabIntent(tabId, binding) {
  const intents = await readTabIntents();
  intents[String(tabId)] = binding;
  await chrome.storage.session.set({ [TAB_INTENTS_KEY]: intents });
}

async function inheritTabIntent(openerTabId, tabId) {
  const intents = await readTabIntents();
  const binding = intents[String(openerTabId)];
  if (!binding) return;
  intents[String(tabId)] = binding;
  await chrome.storage.session.set({ [TAB_INTENTS_KEY]: intents });
}

async function forgetTabIntent(tabId) {
  const intents = await readTabIntents();
  if (!(String(tabId) in intents)) return;
  delete intents[String(tabId)];
  await chrome.storage.session.set({ [TAB_INTENTS_KEY]: intents });
}

/** The live intent bound to the tab a save came from, or null. */
async function tabIntentFor(sender) {
  const tabId = sender.tab?.id;
  if (tabId === undefined) return null;
  const intents = await readTabIntents();
  const binding = intents[String(tabId)];
  if (!binding || !INTENT_PATTERN.test(String(binding.intent_id))) return null;
  if (Number.isFinite(binding.expires_at) && binding.expires_at <= Date.now()) {
    await forgetTabIntent(tabId);
    return null;
  }
  return String(binding.intent_id);
}

/** Wake any open Career Agent page so it fetches the durable event now
 * instead of on its next poll. Best effort: with no page open the event
 * simply waits on the backend. */
async function notifyAppPages(body) {
  if (!body?.conversation_id || !body?.capture_event_id) return;
  let tabs = [];
  try {
    tabs = await chrome.tabs.query({ url: APP_TAB_PATTERNS });
  } catch {
    return;
  }
  await Promise.all(
    tabs.map((tab) =>
      tab.id === undefined
        ? Promise.resolve()
        : chrome.tabs.sendMessage(tab.id, {
            type: "CAREER_AGENT_JOB_CAPTURED",
            conversation_id: body.conversation_id,
            capture_event_id: body.capture_event_id,
          }).catch(() => undefined),
    ),
  );
}

async function saveJob(job, sender) {
  const senderUrl = sender.tab?.url || "";
  if (!isBossUrl(senderUrl) || !job || !isBossUrl(job.source_url)) {
    throw captureError("UNTRUSTED_PAGE", "只能从 BOSS 直聘页面保存岗位");
  }
  const { careerAgentCaptureApiKey } = await chrome.storage.local.get(
    "careerAgentCaptureApiKey",
  );
  if (!careerAgentCaptureApiKey) {
    throw captureError("CAPTURE_CREDENTIAL_MISSING", "请先为 Career Agent 配置扩展密钥，再回来保存");
  }
  const payload = {
    source_url: String(job.source_url || ""),
    title: String(job.title || "").slice(0, 500),
    company_name: String(job.company_name || "").slice(0, 500),
    description: String(job.description || "").slice(0, 100_000),
    city: optional(job.city, 200),
    salary: optional(job.salary, 200),
    experience: optional(job.experience, 200),
    education: optional(job.education, 200),
  };
  if (!payload.title || !payload.company_name || !payload.description) {
    throw captureError("INCOMPLETE_JOB", "当前页面还没有加载出完整岗位详情");
  }
  // Read from the sender's tab, not from anything the page could supply: a
  // job saved from the user's own browsing has no intent and stays a plain
  // library save.
  const intentId = await tabIntentFor(sender);
  if (intentId !== null) payload.capture_intent_id = intentId;

  let lastError = null;
  for (const endpoint of API_ENDPOINTS) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Career-Agent-Capture": "v1",
          Authorization: `Bearer ${careerAgentCaptureApiKey}`,
        },
        body: JSON.stringify(payload),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw captureError("API_REJECTED", body.detail || `保存失败（HTTP ${response.status}）`);
      if (body.capture_event_created) await notifyAppPages(body);
      return body;
    } catch (error) {
      lastError = error;
    }
  }
  if (lastError?.code === "API_REJECTED") throw lastError;
  throw captureError("API_UNREACHABLE", "无法连接本地 Career Agent，请确认后端已启动");
}

/** Report a saved posting the user found closed.
 *
 * Carries only the page URL: what job that is, and whether it is even in the
 * library, is the server's to resolve from what it already stores. Sending an
 * id the page does not have would mean guessing one.
 */
async function reportClosed(sourceUrl, sender) {
  const senderUrl = sender.tab?.url || "";
  if (!isBossUrl(senderUrl) || !isBossUrl(sourceUrl)) {
    throw captureError("UNTRUSTED_PAGE", "只能从 BOSS 直聘页面标记岗位");
  }
  const { careerAgentCaptureApiKey } = await chrome.storage.local.get(
    "careerAgentCaptureApiKey",
  );
  if (!careerAgentCaptureApiKey) {
    throw captureError("CAPTURE_CREDENTIAL_MISSING", "请先为 Career Agent 配置扩展密钥，再回来标记");
  }
  let lastError = null;
  for (const endpoint of CLOSURE_ENDPOINTS) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Career-Agent-Capture": "v1",
          Authorization: `Bearer ${careerAgentCaptureApiKey}`,
        },
        body: JSON.stringify({ source_url: String(sourceUrl) }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw captureError("API_REJECTED", body.detail || `标记失败（HTTP ${response.status}）`);
      }
      return body;
    } catch (error) {
      lastError = error;
    }
  }
  if (lastError?.code === "API_REJECTED") throw lastError;
  throw captureError("API_UNREACHABLE", "无法连接本地 Career Agent，请确认后端已启动");
}

function isAppUrl(value) {
  try {
    return APP_ORIGINS.includes(new URL(value).origin);
  } catch {
    return false;
  }
}

function isBossUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && (url.hostname === "zhipin.com" || url.hostname.endsWith(".zhipin.com"));
  } catch {
    return false;
  }
}

function optional(value, maxLength) {
  const normalized = String(value || "").trim().slice(0, maxLength);
  return normalized || null;
}

function captureError(code, message) {
  const error = new Error(typeof message === "string" ? message : JSON.stringify(message));
  error.code = code;
  return error;
}
