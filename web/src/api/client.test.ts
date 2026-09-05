import { afterEach, describe, expect, it, vi } from "vitest";

import {
  deleteConversation,
  deleteSavedJob,
  fetchEmailWorkspace,
  fetchSavedJobs,
  importResume,
  setJobPursuit,
} from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("workspace client identity boundary", () => {
  it("never submits a user identity on reads", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response("[]", { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchSavedJobs({
      apiBaseUrl: "/api",
      includeIgnored: true,
    });

    const [url] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("/api/v1/jobs?include_dismissed=true");
    expect(String(url)).not.toContain("user_id");
  });

  it("submits only the direct workspace mutation", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response("{}", { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await setJobPursuit("job-1", "dismissed", {
      apiBaseUrl: "/api",
    });

    const request = fetchMock.mock.calls[0][1];
    expect(JSON.parse(String(request?.body))).toEqual({
      pursuit_status: "dismissed",
    });
  });

  it("permanently deletes a saved job without putting identity in the request", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response("{}", { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await deleteSavedJob("job/1", { apiBaseUrl: "/api" });

    const [url, request] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("/api/v1/jobs/job%2F1");
    expect(request?.method).toBe("DELETE");
    expect(request?.body).toBeUndefined();
  });

  it("deletes a conversation using its encoded session id", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response("{}", { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await deleteConversation("conversation/1", { apiBaseUrl: "/api" });

    const [url, request] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("/api/v1/conversations/conversation%2F1");
    expect(request?.method).toBe("DELETE");
  });

  it("reads safe email workspace data without submitting identity", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) => new Response('{"accounts":[],"events":[]}', {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchEmailWorkspace({ apiBaseUrl: "/api" });

    expect(String(fetchMock.mock.calls[0][0])).toBe("/api/v1/email");
  });

  it("uploads a resume as multipart data", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(JSON.stringify({
        resume_id: "r1",
        resume_version_id: "v1",
        name: "主简历",
        version_number: 1,
        document_format: "markdown",
        byte_size: 8,
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await importResume(
      {
        file: new File(["# Resume"], "resume.md", { type: "text/markdown" }),
        name: "主简历",
        targetRoleId: "role-1",
      },
      { apiBaseUrl: "/api" },
    );

    const [url, request] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("/api/v1/resumes/import");
    expect(request?.method).toBe("POST");
    expect(request?.body).toBeInstanceOf(FormData);
  });
});
