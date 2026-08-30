(function syncCareerAgentUser() {
  const STORAGE_KEY = "career-agent:user-id";
  let lastSyncedUserId = "";

  function sync() {
    const userId = window.localStorage.getItem(STORAGE_KEY);
    if (!userId || userId === lastSyncedUserId) return;
    chrome.runtime.sendMessage({
      type: "CAREER_AGENT_SET_USER",
      user_id: userId,
    }).then((response) => {
      if (response?.ok) lastSyncedUserId = userId;
    }).catch(() => undefined);
  }

  sync();
  let attempts = 0;
  const retry = window.setInterval(() => {
    sync();
    attempts += 1;
    if (lastSyncedUserId || attempts >= 10) window.clearInterval(retry);
  }, 500);
  window.addEventListener("focus", sync);
  window.addEventListener("storage", (event) => {
    if (event.key === STORAGE_KEY) sync();
  });
})();
