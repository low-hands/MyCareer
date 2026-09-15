import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  deleteConversation,
  deleteSavedJob,
  fetchApplicationMockInterviews,
  fetchEmailWorkspace,
  fetchReport,
  fetchSavedJobs,
  importResume,
  resumeDocumentUrl,
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

  it("reads mock interview history through the owned application", async () => {
    const fetchMock = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) => new Response(
        '{"application_id":"app/1","title":"Engineer","company_name":"Example","sessions":[]}',
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await fetchApplicationMockInterviews("app/1", { apiBaseUrl: "/api" });

    expect(String(fetchMock.mock.calls[0][0])).toBe(
      "/api/v1/applications/app%2F1/mock-interviews",
    );
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

  it("sends the upload's idempotency key as a header so a retry cannot duplicate the version", async () => {
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

    const result = await importResume(
      {
        file: new File(["# Resume"], "resume.md", { type: "text/markdown" }),
        resumeId: "r1",
        idempotencyKey: "upload-123",
      },
      { apiBaseUrl: "/api" },
    );

    const [, request] = fetchMock.mock.calls[0];
    expect(new Headers(request?.headers).get("Idempotency-Key")).toBe("upload-123");
    expect((request?.body as FormData).get("resume_id")).toBe("r1");
    expect(result).toMatchObject({ resume_id: "r1", resume_version_id: "v1" });
  });

  it("builds ownership-checked document links for viewing and downloading", () => {
    expect(resumeDocumentUrl("r 1", "v/1", { apiBaseUrl: "/api/" })).toBe(
      "/api/v1/resumes/r%201/versions/v%2F1/document",
    );
    expect(resumeDocumentUrl("r1", "v1", { apiBaseUrl: "/api", download: true })).toBe(
      "/api/v1/resumes/r1/versions/v1/document?download=true",
    );
  });

  it("reports a missing delivered body by status so the card can say deleted", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{\"detail\":\"报告不存在。\"}", { status: 404 })),
    );

    const failure = await fetchReport("delivered_body", "body-1", {}, { apiBaseUrl: "/api" })
      .then(() => null, (cause: unknown) => cause);

    expect(failure).toBeInstanceOf(ApiError);
    expect((failure as ApiError).status).toBe(404);
  });
});
