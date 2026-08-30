const API_ENDPOINTS = [
  "http://127.0.0.1:8000/v1/browser-captures/jobs",
  "http://localhost:8000/v1/browser-captures/jobs",
];

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "CAREER_AGENT_SET_USER") {
    const userId = String(message.user_id || "").trim();
    if (!userId || userId.length > 200) {
      sendResponse({ ok: false, code: "INVALID_USER_CONTEXT" });
      return false;
    }
    chrome.storage.local.set({ careerAgentUserId: userId }).then(() => {
      sendResponse({ ok: true });
    });
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

async function saveJob(job, sender) {
  const senderUrl = sender.tab?.url || "";
  if (!isBossUrl(senderUrl) || !job || !isBossUrl(job.source_url)) {
    throw captureError("UNTRUSTED_PAGE", "只能从 BOSS 直聘页面保存岗位");
  }
  const { careerAgentUserId } = await chrome.storage.local.get("careerAgentUserId");
  if (!careerAgentUserId) {
    throw captureError("USER_CONTEXT_MISSING", "请先打开 Career Agent 页面，再回来保存");
  }
  const payload = {
    user_id: careerAgentUserId,
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

  let lastError = null;
  for (const endpoint of API_ENDPOINTS) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Career-Agent-Capture": "v1",
        },
        body: JSON.stringify(payload),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw captureError("API_REJECTED", body.detail || `保存失败（HTTP ${response.status}）`);
      return body;
    } catch (error) {
      lastError = error;
    }
  }
  if (lastError?.code === "API_REJECTED") throw lastError;
  throw captureError("API_UNREACHABLE", "无法连接本地 Career Agent，请确认后端已启动");
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
