export type GoogleOAuthCallback = {
  view: "email" | "calendar";
  status: "connected" | "denied" | "invalid" | "configuration_error" | "network_error" | "internal_error";
  message: string;
};

/** Consume the short-lived OAuth result without retaining it in the URL. */
export function consumeGoogleOAuthCallback(
  location: Pick<Location, "search" | "pathname" | "hash"> = window.location,
  history: Pick<History, "replaceState"> = window.history,
): GoogleOAuthCallback | null {
  const params = new URLSearchParams(location.search);
  const status = params.get("status");
  if (![
    "connected",
    "denied",
    "invalid",
    "configuration_error",
    "network_error",
    "internal_error",
  ].includes(status ?? "")) return null;

  const kind = params.get("kind");
  const view = kind === "calendar" ? "calendar" : "email";
  history.replaceState(null, "", `${location.pathname}${location.hash}`);

  if (status === "connected") {
    return {
      view,
      status: "connected",
      message: kind === "calendar" ? "Google Calendar 已连接。" : "Gmail 已连接。",
    };
  }
  const messages: Record<string, string> = {
    denied: "Google 授权已取消。",
    invalid: "Google 授权已失效或邮箱尚未验证，请重新连接。",
    configuration_error: "Google OAuth 配置不完整，请检查后端配置。",
    network_error: "连接 Google 时网络请求失败，请稍后重试。",
    internal_error: "Google 账号连接失败，请查看后端日志。",
  };
  return {
    view,
    status: status as Exclude<GoogleOAuthCallback["status"], "connected">,
    message: messages[status!] ?? "Google 账号连接失败，请重试。",
  };
}
