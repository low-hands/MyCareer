(function syncCareerAgentCredential() {
  const CAPTURE_KEY_STORAGE_KEY = "career-agent:capture-api-key";
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
})();
