(function mountCareerAgentCapture() {
  const parser = globalThis.CareerAgentBossParser;
  if (!parser || document.getElementById("career-agent-capture-root")) return;

  const host = document.createElement("div");
  host.id = "career-agent-capture-root";
  host.style.position = "fixed";
  host.style.right = "20px";
  host.style.bottom = "20px";
  host.style.zIndex = "2147483647";
  const shadow = host.attachShadow({ mode: "closed" });
  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      .card { width: 300px; box-sizing: border-box; padding: 16px; border: 1px solid #d8ded9; border-radius: 14px; background: #fbfcf8; box-shadow: 0 14px 40px rgba(17, 35, 26, .18); color: #17231d; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      .top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
      .brand { font-size: 12px; font-weight: 700; letter-spacing: .08em; color: #41715a; }
      button { font: inherit; cursor: pointer; }
      .close { border: 0; background: transparent; color: #718078; font-size: 18px; padding: 0 2px; }
      h2 { margin: 10px 0 2px; font-size: 17px; line-height: 1.3; }
      .company { margin: 0; color: #66736c; }
      .meta { display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0; }
      .meta span { padding: 3px 7px; border-radius: 999px; background: #edf1ec; color: #526158; font-size: 12px; }
      .hint { margin: 9px 0 12px; color: #66736c; font-size: 12px; }
      .save { width: 100%; border: 0; border-radius: 9px; padding: 10px 12px; background: #255f43; color: white; font-weight: 700; }
      .save:disabled { cursor: wait; opacity: .65; }
      .report-closed { width: 100%; border: 1px solid #d8ded9; border-radius: 9px; padding: 10px 12px; background: transparent; color: #526158; font-weight: 700; }
      .report-closed:disabled { cursor: wait; opacity: .65; }
      .status { min-height: 18px; margin: 8px 0 0; font-size: 12px; color: #526158; }
      .status.error { color: #a13a32; }
    </style>
    <section class="card" hidden>
      <div class="top"><span class="brand">CAREER AGENT</span><button class="close" type="button" aria-label="暂不保存">×</button></div>
      <h2></h2><p class="company"></p><div class="meta"></div>
      <p class="hint">仅预览当前详情；点击后才会保存 JD，不会自动投递或联系招聘者。</p>
      <button class="save" type="button">保存到 Career Agent</button>
      <button class="report-closed" type="button" hidden>标记为已下架</button>
      <p class="status" role="status"></p>
    </section>`;
  document.documentElement.appendChild(host);

  const card = shadow.querySelector(".card");
  const title = shadow.querySelector("h2");
  const company = shadow.querySelector(".company");
  const meta = shadow.querySelector(".meta");
  const save = shadow.querySelector(".save");
  const reportClosed = shadow.querySelector(".report-closed");
  const hint = shadow.querySelector(".hint");
  const status = shadow.querySelector(".status");
  let currentJob = null;
  let closedUrl = null;
  let dismissedFingerprint = "";
  let lastFingerprint = "";
  let timer = null;

  function fingerprint(job) {
    return `${job.source_url}\u241f${job.title}\u241f${job.company_name}\u241f${job.description.length}`;
  }

  function refresh() {
    timer = null;
    const result = parser.extract(document, window.location);
    // A closed posting has no JD left to capture, so the ordinary card would
    // simply disappear — and the one fact this page still carries, that the
    // job is gone, would go with it. Nothing else in the system can observe
    // that: it is never polled, only seen when the user opens the page.
    if (result.status === "closed") {
      showClosed();
      return;
    }
    if (result.status !== "ready" || !result.job.source_url) {
      currentJob = null;
      closedUrl = null;
      card.hidden = true;
      return;
    }
    reportClosed.hidden = true;
    save.hidden = false;
    closedUrl = null;
    hint.textContent = "仅预览当前详情；点击后才会保存 JD，不会自动投递或联系招聘者。";
    const nextFingerprint = fingerprint(result.job);
    if (nextFingerprint === dismissedFingerprint) return;
    currentJob = result.job;
    card.hidden = false;
    if (nextFingerprint === lastFingerprint) return;
    lastFingerprint = nextFingerprint;
    title.textContent = currentJob.title;
    company.textContent = currentJob.company_name;
    meta.replaceChildren(...[currentJob.salary, currentJob.city, currentJob.experience, currentJob.education]
      .filter(Boolean)
      .map((value) => {
        const chip = document.createElement("span");
        chip.textContent = value;
        return chip;
      }));
    save.disabled = false;
    save.textContent = "保存到 Career Agent";
    status.textContent = "";
    status.classList.remove("error");
  }

  function showClosed() {
    const url = parser.canonicalPageUrl(window.location.href);
    // Only a job detail page. A closed-looking search page or company page is
    // not evidence about any particular posting.
    if (!url || !/\/job_detail\//i.test(url)) {
      card.hidden = true;
      return;
    }
    if (closedUrl === url) return;
    closedUrl = url;
    currentJob = null;
    card.hidden = false;
    title.textContent = "这个岗位已经关闭";
    company.textContent = "";
    meta.replaceChildren();
    hint.textContent = "如果它在你的岗位库里，可以标记为已下架，之后不会再被当成候选。";
    save.hidden = true;
    reportClosed.hidden = false;
    reportClosed.disabled = false;
    reportClosed.textContent = "标记为已下架";
    status.textContent = "";
    status.classList.remove("error");
  }

  reportClosed.addEventListener("click", async () => {
    if (!closedUrl || reportClosed.disabled) return;
    reportClosed.disabled = true;
    reportClosed.textContent = "正在标记…";
    status.textContent = "";
    status.classList.remove("error");
    try {
      const response = await chrome.runtime.sendMessage({
        type: "CAREER_AGENT_REPORT_CLOSED",
        source_url: closedUrl,
      });
      if (!response?.ok) throw new Error(response?.message || "标记失败");
      reportClosed.textContent = "已标记";
      // Said plainly when it matched nothing, rather than reporting a success
      // that changed no record the user has.
      status.textContent = response.result?.matched
        ? "岗位库里的这条已标记为已下架"
        : "这个岗位不在你的岗位库里，没有改动任何记录";
    } catch (error) {
      reportClosed.disabled = false;
      reportClosed.textContent = "重新标记";
      status.textContent = error instanceof Error ? error.message : "标记失败";
      status.classList.add("error");
    }
  });

  function scheduleRefresh() {
    if (timer !== null) return;
    timer = window.setTimeout(refresh, 500);
  }

  shadow.querySelector(".close").addEventListener("click", () => {
    if (currentJob) dismissedFingerprint = fingerprint(currentJob);
    card.hidden = true;
  });
  save.addEventListener("click", async () => {
    if (!currentJob || save.disabled) return;
    save.disabled = true;
    save.textContent = "正在保存…";
    status.textContent = "";
    status.classList.remove("error");
    try {
      const response = await chrome.runtime.sendMessage({
        type: "CAREER_AGENT_SAVE_JOB",
        job: currentJob,
      });
      if (!response?.ok) throw new Error(response?.message || "保存失败");
      save.textContent = "已保存";
      const saved = `JD 快照 v${response.result.snapshot_version} 已进入岗位库`;
      const continuation = response.result.continuation_status;
      const outcome = continuation === "completed"
        ? "已续接原会话"
        : continuation === "pending"
          ? "后端正在续接原会话，关闭页面不影响处理"
          : continuation === "failed"
            ? "续接失败，可在原会话重试"
            : "当前标签未关联对话；岗位已保存，但对话不会自动收到它";
      status.textContent = `${saved}，${outcome}`;
    } catch (error) {
      save.disabled = false;
      save.textContent = "重新保存";
      status.textContent = error instanceof Error ? error.message : "保存失败";
      status.classList.add("error");
    }
  });

  new MutationObserver(scheduleRefresh).observe(document.body, {
    childList: true,
    subtree: true,
  });
  window.addEventListener("popstate", scheduleRefresh);
  refresh();
})();
