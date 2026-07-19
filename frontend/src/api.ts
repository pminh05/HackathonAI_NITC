export interface ClarificationOption {
  option_id: string;
  label: string;
}

export interface ClarificationQuestion {
  question_id: string;
  question_type: "explicit" | "implicit";
  question: string;
  options: ClarificationOption[];
}

export interface ClarificationAnswer {
  question_id: string;
  option_id: string;
  custom_answer?: string;
}

export interface MemorySource {
  memory_id: string;
  score: number;
  categories: string[];
  created_at?: string | null;
  updated_at?: string | null;
}

export interface MemoryCandidate {
  candidate_id: string;
  question_id: string;
  question: string;
  display_value: string;
  options: ClarificationOption[];
  sources: MemorySource[];
}

export interface MemoryConfirmation {
  message?: string;
  category?: string | null;
  candidates: MemoryCandidate[];
}

export interface MemoryConfirmationDecision {
  candidate_id: string;
  action: "use" | "edit" | "ignore";
  option_id?: string;
  custom_answer?: string;
}

export interface MemoryWriteSummary {
  status: "skipped" | "queued" | "failed";
  event_id?: string | null;
}

export interface MemoryWriteStatus {
  status: "skipped" | "pending" | "succeeded" | "failed";
  event_id?: string | null;
  added_count: number;
}

export interface MemoryRecord {
  id: string;
  memory: string;
  categories: string[];
  metadata: {
    thread_id?: string;
    turn_id?: number;
    active_category?: string;
    prompt_version?: string;
  };
  created_at?: string | null;
  updated_at?: string | null;
}

export interface MemoryList {
  count: number;
  page: number;
  page_size: number;
  next?: string | null;
  previous?: string | null;
  results: MemoryRecord[];
}

export interface SelectedProduct {
  product_id: string;
  reason: string;
  trade_off: string;
  name?: string | null;
  image_url?: string | null;
  image_path?: string | null;
  effective_price_vnd?: number | null;
  original_price_vnd?: number | null;
  promotional_price_vnd?: number | null;
}

export interface ThreadStatus {
  thread_id: string;
  status:
    | "running"
    | "waiting_for_clarification"
    | "waiting_for_memory_confirmation"
    | "completed";
  questions: ClarificationQuestion[];
  memory_confirmation?: MemoryConfirmation | null;
  answer: string | null;
  selected_products: SelectedProduct[];
  memory_write?: MemoryWriteSummary | null;
}

export type SseEvent =
  | {
      event: "session";
      data: { thread_id: string; mode: "started" | "continued" | "resumed" };
    }
  | { event: "progress"; data: { stage: string } }
  | {
      event: "clarification_required";
      data: { thread_id: string; message?: string; questions: ClarificationQuestion[] };
    }
  | {
      event: "memory_confirmation_required";
      data: { thread_id: string } & MemoryConfirmation;
    }
  | { event: "token"; data: { delta: string } }
  | {
      event: "completed";
      data: {
        thread_id: string;
        answer: string;
        selected_products: SelectedProduct[];
        memory_write?: MemoryWriteSummary;
      };
    }
  | {
      event: "error";
      data: { code: string; message: string; retryable: boolean };
    };

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(
  /\/$/,
  "",
);

export function resolveProductImageUrl(imagePath: string | null | undefined): string | null {
  const path = imagePath?.trim();
  if (!path) return null;
  if (/^(?:https?:|data:|blob:)/i.test(path)) return path;
  const filename = path.split(/[\\/]/).filter(Boolean).pop();
  return filename
    ? `${apiBaseUrl}/product-images/${encodeURIComponent(filename)}`
    : null;
}

function headers(accessToken: string | null, json = false): HeadersInit {
  return {
    ...(json ? { "Content-Type": "application/json" } : {}),
    ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
  };
}

function extractErrorMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return fallback;
}

async function throwForHttpError(response: Response): Promise<void> {
  if (response.ok) return;
  const fallback = `Yêu cầu thất bại (${response.status}).`;
  try {
    const payload: unknown = await response.json();
    throw new ApiError(extractErrorMessage(payload, fallback), response.status);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(fallback, response.status);
  }
}

function parseEventBlock(block: string): SseEvent | null {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventName = value;
    if (field === "data") dataLines.push(value);
  }
  if (eventName === "message" || dataLines.length === 0) return null;
  const supported = new Set([
    "session",
    "progress",
    "clarification_required",
    "memory_confirmation_required",
    "token",
    "completed",
    "error",
  ]);
  if (!supported.has(eventName)) return null;
  try {
    return { event: eventName, data: JSON.parse(dataLines.join("\n")) } as SseEvent;
  } catch {
    throw new Error("Backend trả về một SSE event không hợp lệ.");
  }
}

async function streamPost(
  path: string,
  body: unknown,
  onEvent: (event: SseEvent) => void,
  accessToken: string | null,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: "POST",
    headers: headers(accessToken, true),
    body: JSON.stringify(body),
    signal,
  });
  await throwForHttpError(response);
  if (!response.body) throw new Error("Trình duyệt không thể đọc response stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary = buffer.search(/\r?\n\r?\n/);
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        const delimiter = buffer.slice(boundary).match(/^\r?\n\r?\n/)?.[0] ?? "\n\n";
        buffer = buffer.slice(boundary + delimiter.length);
        const event = parseEventBlock(block);
        if (event) onEvent(event);
        boundary = buffer.search(/\r?\n\r?\n/);
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const event = parseEventBlock(buffer.trim());
      if (event) onEvent(event);
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // The stream may already be closed normally.
    }
    reader.releaseLock();
  }
}

export function streamChat(
  message: string,
  threadId: string | null,
  onEvent: (event: SseEvent) => void,
  accessToken: string | null,
  signal?: AbortSignal,
): Promise<void> {
  return streamPost(
    "/chat",
    threadId ? { message, thread_id: threadId } : { message },
    onEvent,
    accessToken,
    signal,
  );
}

export function resumeChat(
  threadId: string,
  answers: ClarificationAnswer[],
  onEvent: (event: SseEvent) => void,
  accessToken: string | null,
  signal?: AbortSignal,
): Promise<void> {
  return streamPost(
    `/chat/${encodeURIComponent(threadId)}/resume`,
    { kind: "clarification", answers },
    onEvent,
    accessToken,
    signal,
  );
}

export function resumeMemoryConfirmation(
  threadId: string,
  decisions: MemoryConfirmationDecision[],
  onEvent: (event: SseEvent) => void,
  accessToken: string,
  signal?: AbortSignal,
): Promise<void> {
  return streamPost(
    `/chat/${encodeURIComponent(threadId)}/resume`,
    { kind: "memory_confirmation", decisions },
    onEvent,
    accessToken,
    signal,
  );
}

export async function getThreadStatus(
  threadId: string,
  accessToken: string | null,
  signal?: AbortSignal,
): Promise<ThreadStatus> {
  const response = await fetch(`${apiBaseUrl}/chat/${encodeURIComponent(threadId)}`, {
    headers: headers(accessToken),
    signal,
  });
  await throwForHttpError(response);
  return (await response.json()) as ThreadStatus;
}

export async function getMemoryWriteStatus(
  threadId: string,
  accessToken: string,
  signal?: AbortSignal,
): Promise<MemoryWriteStatus> {
  const response = await fetch(
    `${apiBaseUrl}/chat/${encodeURIComponent(threadId)}/memory-write`,
    { headers: headers(accessToken), signal },
  );
  await throwForHttpError(response);
  return (await response.json()) as MemoryWriteStatus;
}

export async function getMemories(
  accessToken: string,
  signal?: AbortSignal,
): Promise<MemoryList> {
  const response = await fetch(`${apiBaseUrl}/me/memories?page=1&page_size=100`, {
    headers: headers(accessToken),
    signal,
  });
  await throwForHttpError(response);
  return (await response.json()) as MemoryList;
}

export async function deleteMemory(memoryId: string, accessToken: string): Promise<void> {
  const response = await fetch(`${apiBaseUrl}/me/memories/${encodeURIComponent(memoryId)}`, {
    method: "DELETE",
    headers: headers(accessToken),
  });
  await throwForHttpError(response);
}

export async function deleteAllMemories(accessToken: string): Promise<void> {
  const response = await fetch(`${apiBaseUrl}/me/memories`, {
    method: "DELETE",
    headers: headers(accessToken),
  });
  await throwForHttpError(response);
}
