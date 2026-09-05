export const CAPTURE_API_KEY_STORAGE = "career-agent:capture-api-key";

/** Make the extension's separately scoped key visible to its local bridge. */
export function seedCaptureApiKey(): void {
  if (typeof window === "undefined") return;
  const configured = (
    import.meta.env.VITE_CAPTURE_API_KEY as string | undefined
  )?.trim();
  if (configured) window.localStorage.setItem(CAPTURE_API_KEY_STORAGE, configured);
}
