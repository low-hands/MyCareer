(function careerAgentAppBridge() {
  const CAPTURE_KEY_STORAGE_KEY = "career-agent:capture-api-key";
  const OPEN_SEARCH = "career-agent:open-job-search";
  const OPEN_SEARCH_RESULT = "career-agent:open-job-search-result";
  const JOB_CAPTURED = "career-agent:job-captured";
  let lastSyncedApiKey = "";

  function sync() {
    const apiKey = window.localStorage.getItem(CAPTURE_KEY_STORAGE_KEY);
    if (!apiKey || apiKey === lastSyncedApiKey) return;
    chrome.runtime.sendMessage({
      type: "CAREER_AGENT_SET_CAPTURE_CREDENTIAL",
      api_key: apiKey,
    }).then((response) => {
      if (response?.ok) lastSyncedApiKey = apiKey;
    }).catch(() => undefined);
  }

  sync();
  let attempts = 0;
  const retry = window.setInterval(() => {
    sync();
    attempts += 1;
    if (lastSyncedApiKey || attempts >= 10) window.clearInterval(retry);
  }, 500);
  window.addEventListener("focus", sync);
  window.addEventListener("storage", (event) => {
    if (event.key === CAPTURE_KEY_STORAGE_KEY) sync();
  });

  // The page asks the extension to open BOSS so the capture intent can ride
  // along with the *tab*, not the URL. Only this page's own frame is heard:
  // a message from another origin or an embedded frame is ignored.
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== window.location.origin) return;
    const data = event.data;
    if (!data || data.type !== OPEN_SEARCH) return;
    const requestId = String(data.request_id || "");
    chrome.runtime.sendMessage({
      type: "CAREER_AGENT_OPEN_JOB_SEARCH",
      url: String(data.url || ""),
      capture_intent_id: data.capture_intent_id ? String(data.capture_intent_id) : null,
      capture_intent_expires_at: data.capture_intent_expires_at
        ? String(data.capture_intent_expires_at)
        : null,
    }).then((response) => {
      window.postMessage(
        {
          type: OPEN_SEARCH_RESULT,
          request_id: requestId,
          ok: Boolean(response?.ok),
          code: response?.code || null,
        },
        window.location.origin,
      );
    }).catch(() => {
      window.postMessage(
        { type: OPEN_SEARCH_RESULT, request_id: requestId, ok: false, code: "BRIDGE_UNAVAILABLE" },
        window.location.origin,
      );
    });
  });

  // A save landed with a live intent. The page is only nudged: it reads the
  // durable event from the backend itself rather than trusting this payload.
  chrome.runtime.onMessage.addListener((message) => {
    if (message?.type !== "CAREER_AGENT_JOB_CAPTURED") return false;
    window.postMessage(
      {
        type: JOB_CAPTURED,
        conversation_id: String(message.conversation_id || ""),
        capture_event_id: String(message.capture_event_id || ""),
      },
      window.location.origin,
    );
    return false;
  });
})();
