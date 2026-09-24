(function (root, factory) {
  const parser = factory();
  if (typeof module === "object" && module.exports) module.exports = parser;
  if (root) root.CareerAgentBossParser = Object.freeze(parser);
})(typeof globalThis === "object" ? globalThis : this, function () {
  const FIELD_SELECTORS = Object.freeze({
    title: [
      ".job-banner .name",
      ".job-title",
      ".job-name",
      ".job-detail-header h1",
      "h1[class*='job']",
    ],
    company: [
      ".sider-company [ka*='job-detail-company']",
      ".sider-company .company-info .name",
      ".sider-company .company-name",
      ".sider-company .company-info a",
      ".company-card .company-name",
      ".company-name",
      "[class*='company-name']",
      "[class*='brand-name']",
    ],
    salary: [
      ".job-banner .salary",
      ".job-salary",
      ".salary",
      "[class*='salary']",
    ],
    description: [
      ".job-detail-section .job-sec-text",
      ".job-detail-section .text",
      ".job-sec .job-sec-text",
      ".job-sec-text",
      ".job-description",
      "[class*='job-description']",
    ],
    metadata: [
      ".job-banner",
      ".job-primary",
      ".job-detail-header",
      "[class*='job-banner']",
    ],
  });

  function compact(value) {
    return String(value || "").replace(/[\t\f\v ]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  }

  function oneLine(value) {
    return compact(value).replace(/\s+/g, " ");
  }

  function isVisible(node) {
    if (!node || node.hidden || node.getAttribute?.("aria-hidden") === "true") return false;
    if (node.closest?.("[hidden], [aria-hidden='true']")) return false;

    const style = node.ownerDocument?.defaultView?.getComputedStyle?.(node);
    if (style && (style.display === "none" || style.visibility === "hidden")) return false;

    // BOSS is a client-side app. During navigation it can leave the previous
    // detail page mounted but hidden for a short time. querySelector() then
    // returns that stale node, mixing the old company with the new title/JD.
    if (typeof node.getClientRects === "function" && node.getClientRects().length === 0) return false;
    return true;
  }

  function firstText(documentRef, selectors) {
    for (const selector of selectors) {
      const nodes = typeof documentRef.querySelectorAll === "function"
        ? documentRef.querySelectorAll(selector)
        : [documentRef.querySelector(selector)].filter(Boolean);
      for (const node of nodes) {
        if (!isVisible(node)) continue;
        const value = oneLine(node?.innerText || node?.textContent || "");
        if (value) return value;
      }
    }
    return "";
  }

  const JOB_SECTION_LABEL = /(?:职位描述|岗位职责|工作职责|工作内容|任职要求|岗位要求|任职资格)/;
  const COMPANY_SECTION_LABEL = /(?:公司介绍|公司基本信息|企业介绍|工商信息|公司简介)/;

  function descriptionSectionScore(node, value) {
    const scope = node.closest?.(
      ".job-detail-section, .job-sec, section, [class*='detail-section']",
    );
    const heading = scope?.querySelector?.(
      "h1, h2, h3, h4, .title, [class*='section-title']",
    );
    const headingText = oneLine(heading?.innerText || heading?.textContent || "");
    const opening = oneLine(scope?.innerText || scope?.textContent || value).slice(0, 120);
    const labelText = `${headingText} ${opening}`;
    if (COMPANY_SECTION_LABEL.test(labelText) && !JOB_SECTION_LABEL.test(labelText)) return -10_000;
    if (JOB_SECTION_LABEL.test(labelText)) return 10_000 + Math.min(value.length, 5_000);
    return Math.min(value.length, 5_000);
  }

  function bodyDescription(documentRef) {
    const bodyText = compact(documentRef.body?.innerText || "");
    const start = bodyText.search(JOB_SECTION_LABEL);
    if (start < 0) return "";
    let section = bodyText.slice(start, start + 30_000);
    const tail = section.slice(40);
    const end = tail.search(new RegExp(`\\n${COMPANY_SECTION_LABEL.source}`));
    if (end >= 0) section = section.slice(0, end + 40);
    return compact(section).slice(0, 100_000);
  }

  function bestDescription(documentRef) {
    const candidates = [];
    for (const selector of FIELD_SELECTORS.description) {
      for (const node of documentRef.querySelectorAll(selector)) {
        if (!isVisible(node)) continue;
        const value = compact(node?.innerText || node?.textContent || "");
        if (value.length >= 20 && !candidates.some((item) => item.value === value)) {
          candidates.push({ value, score: descriptionSectionScore(node, value) });
        }
      }
    }
    const labeled = candidates
      .filter((item) => item.score >= 10_000)
      .sort((left, right) => right.score - left.score);
    if (labeled[0]) return labeled[0].value.slice(0, 100_000);

    const fromBody = bodyDescription(documentRef);
    if (fromBody) return fromBody;

    const unlabeled = candidates
      .filter((item) => item.score >= 0)
      .sort((left, right) => right.score - left.score);
    return unlabeled[0]?.value.slice(0, 100_000) || "";
  }

  function pageBarrier(documentRef) {
    const bodyText = oneLine(documentRef.body?.innerText || "").slice(0, 5_000);
    if (/安全验证|请完成安全验证|请滑动|访问行为异常|账号存在风险/i.test(bodyText)) return "security_check";
    if (/请登录|登录后查看|立即登录|扫码登录/i.test(bodyText)) return "login_required";
    if (closedPosting(documentRef)) return "closed";
    return null;
  }

  // Read as a barrier because it is one: there is no JD left to capture. It
  // is checked last so a login wall or a captcha never reads as a closure —
  // a page we were not allowed to see says nothing about whether the job is
  // still open, and recording "closed" from it would be inventing an
  // employer decision out of our own lack of access.
  function closedPosting(documentRef) {
    const bodyText = oneLine(documentRef.body?.innerText || "").slice(0, 5_000);
    return /职位已关闭|该职位已下线|职位已下架|职位不存在|该职位已失效|岗位已关闭/i.test(bodyText);
  }

  function firstMatch(value, pattern) {
    return oneLine(String(value || "").match(pattern)?.[0] || "");
  }

  function canonicalPageUrl(rawUrl) {
    try {
      const url = new URL(rawUrl);
      if (url.protocol !== "https:" || !(url.hostname === "zhipin.com" || url.hostname.endsWith(".zhipin.com"))) return "";
      return `${url.protocol}//${url.hostname}${url.pathname}`;
    } catch {
      return "";
    }
  }

  function extract(documentRef, locationRef) {
    const barrier = pageBarrier(documentRef);
    if (barrier) return { status: barrier };

    const metadata = FIELD_SELECTORS.metadata
      .map((selector) => firstText(documentRef, [selector]))
      .filter(Boolean)
      .join(" ");
    const title = firstText(documentRef, FIELD_SELECTORS.title);
    const companyName = firstText(documentRef, FIELD_SELECTORS.company);
    const description = bestDescription(documentRef);
    if (!title || !companyName || !description) {
      return {
        status: "not_ready",
        missing_fields: [
          ["title", title],
          ["company_name", companyName],
          ["description", description],
        ].filter((entry) => !entry[1]).map((entry) => entry[0]),
      };
    }

    return {
      status: "ready",
      job: {
        source_url: canonicalPageUrl(locationRef.href),
        title: title.slice(0, 500),
        company_name: companyName.slice(0, 500),
        description,
        salary: firstText(documentRef, FIELD_SELECTORS.salary) || firstMatch(metadata, /(?:\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?[Kk](?:·\d+薪)?|面议)/),
        city: firstMatch(metadata, /(北京|上海|广州|深圳|杭州|成都|武汉|南京|苏州|西安|长沙|重庆|天津|郑州|厦门|合肥|佛山|东莞|珠海|中山|全国|远程)(?:·[^\s，,。]{1,12})?/),
        experience: firstMatch(metadata, /(经验不限|不限经验|在校\/应届|应届|1年以内|\d+-\d+年|\d+年以内|\d+年以上)/),
        education: firstMatch(metadata, /(学历不限|本科|大专|硕士|博士|高中|中专)/),
      },
    };
  }

  return { canonicalPageUrl, closedPosting, extract, pageBarrier };
});
