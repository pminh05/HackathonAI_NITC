import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ApiError,
  ClarificationAnswer,
  ClarificationQuestion,
  MemoryCandidate,
  MemoryConfirmation,
  MemoryConfirmationDecision,
  MemoryRecord,
  MemoryWriteSummary,
  SelectedProduct,
  SseEvent,
  deleteAllMemories,
  deleteMemory,
  getMemories,
  getMemoryWriteStatus,
  getThreadStatus,
  resolveProductImageUrl,
  resumeChat,
  resumeMemoryConfirmation,
  streamChat,
} from "./api";
import { authConfigured, supabase, type Session } from "./auth";

const storageKeyFor = (userId: string | null): string =>
  `product-advisor-chat-v2:${userId || "anonymous"}`;

type UserItem = { id: string; type: "user"; text: string };
type AssistantItem = {
  id: string;
  type: "assistant";
  text: string;
  products: SelectedProduct[];
  streaming?: boolean;
};
type ClarificationItem = {
  id: string;
  type: "clarification";
  questions: ClarificationQuestion[];
  answers: Record<string, ClarificationAnswer>;
  confirmedIds: string[];
  submitted: boolean;
};
type MemoryConfirmationItem = {
  id: string;
  type: "memory_confirmation";
  confirmation: MemoryConfirmation;
  submitted: boolean;
};
type TimelineItem =
  | UserItem
  | AssistantItem
  | ClarificationItem
  | MemoryConfirmationItem;
type Phase =
  | "idle"
  | "sending"
  | "waiting"
  | "resuming"
  | "restoring"
  | "error";

interface ChatState {
  threadId: string | null;
  items: TimelineItem[];
  phase: Phase;
  progress: string | null;
  error: string | null;
}

type Action =
  | { type: "LOAD_STATE"; state: ChatState }
  | { type: "ADD_USER"; item: UserItem }
  | { type: "ADD_ASSISTANT"; item: AssistantItem }
  | { type: "SET_THREAD"; threadId: string }
  | { type: "SET_PHASE"; phase: Phase; progress?: string | null }
  | { type: "SET_PROGRESS"; progress: string | null }
  | { type: "APPEND_TOKEN"; id: string; delta: string }
  | {
      type: "COMPLETE";
      id: string;
      answer: string;
      products: SelectedProduct[];
    }
  | {
      type: "ADD_CLARIFICATION";
      assistantId: string;
      questions: ClarificationQuestion[];
    }
  | {
      type: "ADD_MEMORY_CONFIRMATION";
      assistantId: string;
      confirmation: MemoryConfirmation;
    }
  | {
      type: "SELECT_OPTION";
      itemId: string;
      questionId: string;
      optionId: string;
    }
  | {
      type: "SET_CUSTOM_ANSWER";
      itemId: string;
      questionId: string;
      value: string;
    }
  | { type: "CONFIRM_ANSWER"; itemId: string; questionId: string }
  | { type: "MARK_SUBMITTED"; itemId: string }
  | { type: "MARK_MEMORY_SUBMITTED"; itemId: string }
  | { type: "RECONCILE_WAITING"; questions: ClarificationQuestion[] }
  | { type: "RECONCILE_MEMORY"; confirmation: MemoryConfirmation }
  | { type: "RECONCILE_COMPLETED"; answer: string; products: SelectedProduct[] }
  | { type: "REQUEST_FAILED"; assistantId?: string; message: string }
  | { type: "CLEAR_ERROR" }
  | { type: "RESET"; message?: string };

const emptyState: ChatState = {
  threadId: null,
  items: [],
  phase: "idle",
  progress: null,
  error: null,
};

function createId(): string {
  return crypto.randomUUID();
}

function loadState(storageKey: string): ChatState {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return emptyState;
    const saved = JSON.parse(raw) as {
      version?: number;
      threadId?: unknown;
      items?: unknown;
    };
    if (
      saved.version !== 2 ||
      typeof saved.threadId !== "string" ||
      !Array.isArray(saved.items)
    ) {
      return emptyState;
    }
    return {
      threadId: saved.threadId,
      items: saved.items as TimelineItem[],
      phase: "restoring",
      progress: "Đang khôi phục cuộc trò chuyện…",
      error: null,
    };
  } catch {
    localStorage.removeItem(storageKey);
    return emptyState;
  }
}

function validStoredAnswer(
  question: ClarificationQuestion,
  answer: ClarificationAnswer | undefined,
): answer is ClarificationAnswer {
  if (!answer || answer.question_id !== question.question_id) return false;
  if (!question.options.some((option) => option.option_id === answer.option_id))
    return false;
  return answer.option_id !== "other" || Boolean(answer.custom_answer?.trim());
}

function reconcileClarification(
  existing: ClarificationItem | undefined,
  questions: ClarificationQuestion[],
): ClarificationItem {
  const answers: Record<string, ClarificationAnswer> = {};
  const confirmedIds: string[] = [];

  for (const question of questions) {
    const answer = existing?.answers[question.question_id];
    if (validStoredAnswer(question, answer))
      answers[question.question_id] = answer;
  }
  for (const question of questions) {
    if (
      !existing?.confirmedIds.includes(question.question_id) ||
      !answers[question.question_id]
    )
      break;
    confirmedIds.push(question.question_id);
  }

  return {
    id: existing?.id ?? createId(),
    type: "clarification",
    questions,
    answers,
    confirmedIds,
    submitted: false,
  };
}

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "LOAD_STATE":
      return action.state;
    case "ADD_USER":
      return { ...state, items: [...state.items, action.item], error: null };
    case "ADD_ASSISTANT":
      return { ...state, items: [...state.items, action.item] };
    case "SET_THREAD":
      return { ...state, threadId: action.threadId };
    case "SET_PHASE":
      return {
        ...state,
        phase: action.phase,
        progress:
          action.progress === undefined ? state.progress : action.progress,
        error: action.phase === "error" ? state.error : null,
      };
    case "SET_PROGRESS":
      return { ...state, progress: action.progress };
    case "APPEND_TOKEN":
      return {
        ...state,
        progress: null,
        items: state.items.map((item) =>
          item.id === action.id && item.type === "assistant"
            ? { ...item, text: item.text + action.delta }
            : item,
        ),
      };
    case "COMPLETE":
      return {
        ...state,
        phase: "idle",
        progress: null,
        error: null,
        items: state.items.map((item) =>
          item.id === action.id && item.type === "assistant"
            ? {
                ...item,
                text: action.answer,
                products: action.products.slice(0, 3),
                streaming: false,
              }
            : item,
        ),
      };
    case "ADD_CLARIFICATION":
      return {
        ...state,
        phase: "waiting",
        progress: null,
        error: null,
        items: [
          ...state.items.filter((item) => item.id !== action.assistantId),
          reconcileClarification(undefined, action.questions),
        ],
      };
    case "ADD_MEMORY_CONFIRMATION":
      return {
        ...state,
        phase: "waiting",
        progress: null,
        error: null,
        items: [
          ...state.items.filter((item) => item.id !== action.assistantId),
          {
            id: createId(),
            type: "memory_confirmation",
            confirmation: action.confirmation,
            submitted: false,
          },
        ],
      };
    case "SELECT_OPTION":
      return {
        ...state,
        items: state.items.map((item) => {
          if (item.id !== action.itemId || item.type !== "clarification")
            return item;
          return {
            ...item,
            answers: {
              ...item.answers,
              [action.questionId]: {
                question_id: action.questionId,
                option_id: action.optionId,
              },
            },
          };
        }),
      };
    case "SET_CUSTOM_ANSWER":
      return {
        ...state,
        items: state.items.map((item) => {
          if (item.id !== action.itemId || item.type !== "clarification")
            return item;
          const current = item.answers[action.questionId];
          if (!current) return item;
          return {
            ...item,
            answers: {
              ...item.answers,
              [action.questionId]: { ...current, custom_answer: action.value },
            },
          };
        }),
      };
    case "CONFIRM_ANSWER":
      return {
        ...state,
        items: state.items.map((item) =>
          item.id === action.itemId && item.type === "clarification"
            ? {
                ...item,
                confirmedIds: item.confirmedIds.includes(action.questionId)
                  ? item.confirmedIds
                  : [...item.confirmedIds, action.questionId],
              }
            : item,
        ),
      };
    case "MARK_SUBMITTED":
      return {
        ...state,
        items: state.items.map((item) =>
          item.id === action.itemId && item.type === "clarification"
            ? {
                ...item,
                confirmedIds: item.questions
                  .filter((question) =>
                    validStoredAnswer(
                      question,
                      item.answers[question.question_id],
                    ),
                  )
                  .map((question) => question.question_id),
                submitted: true,
              }
            : item,
        ),
      };
    case "MARK_MEMORY_SUBMITTED":
      return {
        ...state,
        items: state.items.map((item) =>
          item.id === action.itemId && item.type === "memory_confirmation"
            ? { ...item, submitted: true }
            : item,
        ),
      };
    case "RECONCILE_WAITING": {
      const existingIndex = state.items.findLastIndex(
        (item) => item.type === "clarification",
      );
      const lastUserIndex = state.items.findLastIndex(
        (item) => item.type === "user",
      );
      const existing =
        existingIndex > lastUserIndex
          ? (state.items[existingIndex] as ClarificationItem)
          : undefined;
      const clarification = reconcileClarification(existing, action.questions);
      const items = state.items.filter(
        (item) => !(item.type === "assistant" && item.streaming && !item.text),
      );
      if (existing) {
        const currentIndex = items.findIndex(
          (item) => item.id === existing?.id,
        );
        if (currentIndex >= 0) items[currentIndex] = clarification;
        else items.push(clarification);
      } else {
        items.push(clarification);
      }
      return { ...state, items, phase: "waiting", progress: null, error: null };
    }
    case "RECONCILE_MEMORY": {
      const existingIndex = state.items.findLastIndex(
        (item) => item.type === "memory_confirmation",
      );
      const lastUserIndex = state.items.findLastIndex(
        (item) => item.type === "user",
      );
      const existing =
        existingIndex > lastUserIndex
          ? (state.items[existingIndex] as MemoryConfirmationItem)
          : undefined;
      const memoryItem: MemoryConfirmationItem = {
        id: existing?.id ?? createId(),
        type: "memory_confirmation",
        confirmation: action.confirmation,
        submitted: false,
      };
      const items = state.items.filter(
        (item) => !(item.type === "assistant" && item.streaming && !item.text),
      );
      if (existing) {
        const index = items.findIndex((item) => item.id === existing.id);
        if (index >= 0) items[index] = memoryItem;
        else items.push(memoryItem);
      } else {
        items.push(memoryItem);
      }
      return { ...state, items, phase: "waiting", progress: null, error: null };
    }
    case "RECONCILE_COMPLETED": {
      const products = action.products.slice(0, 3);
      const items = state.items.filter(
        (item) => !(item.type === "assistant" && item.streaming && !item.text),
      );
      const lastUser = items.findLastIndex((item) => item.type === "user");
      const lastClarification = items.findLastIndex(
        (item) => item.type === "clarification",
      );
      const boundary = Math.max(lastUser, lastClarification);
      const lastAssistant = items.findLastIndex(
        (item) => item.type === "assistant",
      );
      if (lastAssistant > boundary) {
        items[lastAssistant] = {
          ...(items[lastAssistant] as AssistantItem),
          text: action.answer,
          products,
          streaming: false,
        };
      } else {
        items.push({
          id: createId(),
          type: "assistant",
          text: action.answer,
          products,
        });
      }
      return { ...state, items, phase: "idle", progress: null, error: null };
    }
    case "REQUEST_FAILED":
      return {
        ...state,
        phase: "error",
        progress: null,
        error: action.message,
        items: action.assistantId
          ? state.items
              .filter(
                (item) =>
                  item.id !== action.assistantId ||
                  item.type !== "assistant" ||
                  Boolean(item.text),
              )
              .map((item) =>
                item.id === action.assistantId && item.type === "assistant"
                  ? { ...item, streaming: false }
                  : item,
              )
          : state.items,
      };
    case "CLEAR_ERROR":
      return { ...state, phase: "idle", error: null };
    case "RESET":
      return { ...emptyState, error: action.message ?? null };
  }
}

const progressLabels: Record<string, string> = {
  intent_detected: "Đang hiểu nhu cầu…",
  memory_recalled: "Đang tìm thông tin từ các phiên trước…",
  memory_projected: "Đang đối chiếu hồ sơ đã nhớ…",
  memory_confirmed: "Đã xác nhận thông tin từ hồ sơ…",
  need_extracted: "Đang hiểu nhu cầu…",
  clarification_ready: "Đang chuẩn bị câu hỏi…",
  clarification_completed: "Đang tìm sản phẩm…",
  filter_built: "Đang tìm sản phẩm…",
  retrieval_completed: "Đang tìm sản phẩm…",
  ranking_completed: "Đang hoàn thiện gợi ý…",
  memory_write_queued: "Đang cập nhật hồ sơ đã nhớ…",
};

const MAX_MESSAGE_WORDS = 1_000;
const starterSuggestions = [
  "Tôi muốn mua một chiếc máy lạnh tốt",
  "Tủ lạnh cho gia đình 4 người",
  // "Tôi muốn biết về chính sách đổi trả",
  // "Tôi muốn biết về chính sách vận chuyển",
];
const followUpSuggestions = [
  // "Tôi muốn biết về chính sách bảo hành",
  // "Tôi muốn biết về chính sách đổi trả",
  // "Tôi muốn biết về chính sách vận chuyển",
  "Tôi muốn biết thêm về sản phẩm",
  "Tôi muốn biết thêm về giá cả",
];

function countWords(value: string): number {
  return value.trim().match(/\S+/g)?.length ?? 0;
}

function limitWords(value: string, maximum: number): string {
  const matches = [...value.matchAll(/\S+/g)];
  if (matches.length <= maximum) return value;
  const lastWord = matches[maximum - 1];
  return value.slice(0, (lastWord.index ?? 0) + lastWord[0].length);
}

function messageFromError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401)
      return "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.";
    if (error.status === 409)
      return "Cuộc trò chuyện đang được xử lý hoặc chưa sẵn sàng.";
    if (error.status === 404)
      return "Không tìm thấy cuộc trò chuyện này trên server.";
    if (error.status === 422) return error.message;
  }
  if (error instanceof Error && error.name === "AbortError")
    return "Yêu cầu đã bị hủy.";
  return error instanceof Error
    ? error.message
    : "Không thể kết nối tới Product Advisor.";
}

function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

function formatPrice(value: number): string {
  return new Intl.NumberFormat("vi-VN", {
    style: "currency",
    currency: "VND",
    maximumFractionDigits: 0,
  }).format(value);
}

function finitePrice(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function MarkdownText({ text }: { text: string }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

function ProductCard({ product }: { product: SelectedProduct }) {
  const promotional = finitePrice(product.promotional_price_vnd)
    ? product.promotional_price_vnd
    : null;
  const original = finitePrice(product.original_price_vnd)
    ? product.original_price_vnd
    : null;
  const effective = finitePrice(product.effective_price_vnd)
    ? product.effective_price_vnd
    : null;
  const currentPrice = promotional ?? effective ?? original;
  const image = product.image_url || product.image_path;

  const imageUrl = image ? image.replace(/^\/public/, "") : undefined;

  return (
    <article className="product-card">
      {imageUrl ? (
        <img
          className="product-image"
          src={imageUrl}
          alt={product.name || product.product_id}
          onError={(event) => {
            event.currentTarget.hidden = true;
          }}
        />
      ) : null}
      <div className="product-content">
        <div className="product-heading">
          <h3>{product.name || product.product_id}</h3>
          {product.name ? (
            <span className="product-id">{product.product_id}</span>
          ) : null}
        </div>
        {currentPrice !== null ? (
          <div className="product-price">
            <strong>{formatPrice(currentPrice)}</strong>
            {promotional !== null &&
            original !== null &&
            original !== promotional ? (
              <del>{formatPrice(original)}</del>
            ) : null}
          </div>
        ) : null}
        {product.reason ? (
          <p>
            <span>Phù hợp:</span> {product.reason}
          </p>
        ) : null}
        {product.trade_off ? (
          <p className="trade-off">
            <span>Đánh đổi:</span> {product.trade_off}
          </p>
        ) : null}
      </div>
    </article>
  );
}

function ClarificationCard({
  item,
  disabled,
  onSelect,
  onCustomAnswer,
  onConfirm,
  onSubmit,
}: {
  item: ClarificationItem;
  disabled: boolean;
  onSelect: (questionId: string, optionId: string) => void;
  onCustomAnswer: (questionId: string, value: string) => void;
  onConfirm: (questionId: string) => void;
  onSubmit: (answers: ClarificationAnswer[]) => void;
}) {
  const activeIndex = item.confirmedIds.length;
  const activeQuestion = item.questions[activeIndex];

  const answerText = (question: ClarificationQuestion): string => {
    const answer = item.answers[question.question_id];
    if (!answer) return "";
    if (answer.option_id === "other") return answer.custom_answer?.trim() || "";
    return (
      question.options.find((option) => option.option_id === answer.option_id)
        ?.label || ""
    );
  };

  const currentAnswer = activeQuestion
    ? item.answers[activeQuestion.question_id]
    : undefined;
  const currentValid = Boolean(
    currentAnswer &&
    (currentAnswer.option_id !== "other" ||
      currentAnswer.custom_answer?.trim()),
  );

  const answersWith = (answer: ClarificationAnswer): ClarificationAnswer[] =>
    item.questions
      .map((question) =>
        question.question_id === answer.question_id
          ? answer
          : item.answers[question.question_id],
      )
      .filter((value): value is ClarificationAnswer => Boolean(value));

  const selectOption = (questionId: string, optionId: string) => {
    const answer: ClarificationAnswer = {
      question_id: questionId,
      option_id: optionId,
    };
    onSelect(questionId, optionId);
    if (optionId === "other") return;
    if (activeIndex === item.questions.length - 1) {
      onSubmit(answersWith(answer));
    } else {
      onConfirm(questionId);
    }
  };

  const confirmCustomAnswer = () => {
    if (!activeQuestion || !currentAnswer || !currentValid) return;
    if (activeIndex === item.questions.length - 1) {
      onSubmit(answersWith(currentAnswer));
    } else {
      onConfirm(activeQuestion.question_id);
    }
  };

  return (
    <section className="clarification-card" aria-label="Câu hỏi làm rõ">
      <div className="advisor-label">Product Advisor</div>
      <h2>Mình cần thêm một chút thông tin</h2>
      {item.questions.slice(0, activeIndex).map((question) => (
        <div className="confirmed-answer" key={question.question_id}>
          <span aria-hidden="true">✓</span>
          <div>
            <strong>{question.question}</strong> — {answerText(question)}
          </div>
        </div>
      ))}

      {activeQuestion && !item.submitted ? (
        <fieldset className="question-fieldset" disabled={disabled}>
          <legend>{activeQuestion.question}</legend>
          <div className="question-count">
            Câu {activeIndex + 1}/{item.questions.length}
          </div>
          <div className="option-list">
            {activeQuestion.options.map((option) => (
              <label
                className={`option-row ${currentAnswer?.option_id === option.option_id ? "selected" : ""}`}
                key={option.option_id}
              >
                <input
                  type="radio"
                  name={activeQuestion.question_id}
                  value={option.option_id}
                  checked={currentAnswer?.option_id === option.option_id}
                  onChange={() =>
                    selectOption(activeQuestion.question_id, option.option_id)
                  }
                />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
          {currentAnswer?.option_id === "other" ? (
            <label className="custom-answer">
              Câu trả lời của bạn
              <input
                type="text"
                value={currentAnswer.custom_answer || ""}
                onChange={(event) =>
                  onCustomAnswer(activeQuestion.question_id, event.target.value)
                }
                placeholder="Nhập câu trả lời…"
                autoFocus
              />
            </label>
          ) : null}
          {currentAnswer?.option_id === "other" ? (
            <button
              className="primary-button question-next"
              type="button"
              disabled={!currentValid || disabled}
              onClick={confirmCustomAnswer}
            >
              Xác nhận
            </button>
          ) : null}
        </fieldset>
      ) : null}
      {item.submitted ? (
        <div className="submitted-note">✓ Đã gửi câu trả lời</div>
      ) : null}
    </section>
  );
}

function MemoryConfirmationCard({
  item,
  disabled,
  onSubmit,
}: {
  item: MemoryConfirmationItem;
  disabled: boolean;
  onSubmit: (decisions: MemoryConfirmationDecision[]) => void;
}) {
  const [decisions, setDecisions] = useState<
    Record<string, MemoryConfirmationDecision>
  >({});

  const choose = (
    candidate: MemoryCandidate,
    action: MemoryConfirmationDecision["action"],
  ) => {
    setDecisions((current) => ({
      ...current,
      [candidate.candidate_id]: {
        candidate_id: candidate.candidate_id,
        action,
      },
    }));
  };

  const setEditedOption = (candidate: MemoryCandidate, optionId: string) => {
    setDecisions((current) => ({
      ...current,
      [candidate.candidate_id]: {
        candidate_id: candidate.candidate_id,
        action: "edit",
        option_id: optionId,
      },
    }));
  };

  const setCustom = (candidate: MemoryCandidate, value: string) => {
    setDecisions((current) => ({
      ...current,
      [candidate.candidate_id]: {
        ...(current[candidate.candidate_id] || {
          candidate_id: candidate.candidate_id,
          action: "edit" as const,
          option_id: "other",
        }),
        custom_answer: value,
      },
    }));
  };

  const ready = item.confirmation.candidates.every((candidate) => {
    const decision = decisions[candidate.candidate_id];
    if (!decision) return false;
    if (decision.action !== "edit") return true;
    if (!decision.option_id) return false;
    return decision.option_id !== "other" || Boolean(decision.custom_answer?.trim());
  });

  const submit = () => {
    if (!ready) return;
    onSubmit(
      item.confirmation.candidates.map(
        (candidate) => decisions[candidate.candidate_id],
      ),
    );
  };

  return (
    <section className="memory-confirmation-card" aria-label="Xác nhận hồ sơ đã nhớ">
      <div className="advisor-label">Từ các phiên trước</div>
      <h2>Mình nhớ một vài điều về bạn</h2>
      <p className="memory-confirmation-intro">
        {item.confirmation.message ||
          "Chọn thông tin bạn muốn dùng cho lần tư vấn này."}
      </p>
      <div className="memory-candidate-list">
        {item.confirmation.candidates.map((candidate) => {
          const decision = decisions[candidate.candidate_id];
          return (
            <article className="memory-candidate" key={candidate.candidate_id}>
              <span className="memory-source">Hồ sơ đã nhớ</span>
              <strong>{candidate.display_value}</strong>
              <small>{candidate.question}</small>
              <div className="memory-actions" role="group">
                <button
                  type="button"
                  className={decision?.action === "use" ? "selected" : ""}
                  disabled={disabled || item.submitted}
                  onClick={() => choose(candidate, "use")}
                >
                  Dùng
                </button>
                <button
                  type="button"
                  className={decision?.action === "edit" ? "selected" : ""}
                  disabled={disabled || item.submitted}
                  onClick={() => choose(candidate, "edit")}
                >
                  Sửa
                </button>
                <button
                  type="button"
                  className={decision?.action === "ignore" ? "selected" : ""}
                  disabled={disabled || item.submitted}
                  onClick={() => choose(candidate, "ignore")}
                >
                  Không dùng
                </button>
              </div>
              {decision?.action === "edit" ? (
                <div className="memory-edit-fields">
                  <select
                    value={decision.option_id || ""}
                    disabled={disabled || item.submitted}
                    onChange={(event) =>
                      setEditedOption(candidate, event.target.value)
                    }
                  >
                    <option value="">Chọn giá trị mới…</option>
                    {candidate.options.map((option) => (
                      <option value={option.option_id} key={option.option_id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {decision.option_id === "other" ? (
                    <input
                      type="text"
                      value={decision.custom_answer || ""}
                      disabled={disabled || item.submitted}
                      placeholder="Nhập câu trả lời khác…"
                      onChange={(event) => setCustom(candidate, event.target.value)}
                    />
                  ) : null}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
      {item.submitted ? (
        <div className="submitted-note">✓ Đã gửi lựa chọn</div>
      ) : (
        <button
          type="button"
          className="primary-button submit-memory"
          disabled={disabled || !ready}
          onClick={submit}
        >
          Xác nhận và tiếp tục
        </button>
      )}
    </section>
  );
}

const memoryCategoryLabels: Record<string, string> = {
  identity_style: "Tên gọi và phong cách",
  household_context: "Gia đình và sinh hoạt",
  shopping_preference: "Ưu tiên mua sắm",
  category_need: "Nhu cầu theo ngành hàng",
  product_interaction: "Sản phẩm đã tương tác",
  feedback: "Phản hồi sản phẩm",
};

const productCategoryLabels: Record<string, string> = {
  refrigerator: "Tủ lạnh",
  washing_machine: "Máy giặt",
  air_conditioner: "Máy lạnh",
  dryer: "Máy sấy quần áo",
  dishwasher: "Máy rửa chén",
  cooler_freezer: "Tủ mát, tủ đông",
  water_heater: "Máy nước nóng",
};

function MemoryPanel({
  records,
  count,
  loading,
  error,
  onClose,
  onRefresh,
  onDelete,
  onDeleteAll,
}: {
  records: MemoryRecord[];
  count: number;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onRefresh: () => void;
  onDelete: (id: string) => void;
  onDeleteAll: () => void;
}) {
  const groups = records.reduce<Record<string, MemoryRecord[]>>((result, record) => {
    const category = record.categories[0] || "Thông tin khác";
    (result[category] ||= []).push(record);
    return result;
  }, {});
  const formatTime = (value?: string | null) => {
    if (!value) return "Không rõ thời gian";
    const date = new Date(value);
    return Number.isNaN(date.getTime())
      ? "Không rõ thời gian"
      : new Intl.DateTimeFormat("vi-VN", {
          dateStyle: "short",
          timeStyle: "short",
        }).format(date);
  };

  return (
    <div className="memory-panel-backdrop" role="presentation" onMouseDown={onClose}>
      <aside
        className="memory-panel"
        aria-label="Hồ sơ đang nhớ"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="memory-panel-heading">
          <div>
            <span>Cá nhân hóa xuyên phiên</span>
            <h2>Hồ sơ đang nhớ ({count})</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Đóng hồ sơ">
            ×
          </button>
        </div>
        {loading ? <p className="memory-panel-state">Đang tải hồ sơ…</p> : null}
        {error ? (
          <div className="memory-panel-error">
            <span>{error}</span>
            <button type="button" onClick={onRefresh}>Thử lại</button>
          </div>
        ) : null}
        {!loading && !error && records.length === 0 ? (
          <p className="memory-panel-state">Chưa có thông tin nào được ghi nhớ.</p>
        ) : null}
        <div className="memory-groups">
          {Object.entries(groups).map(([category, items]) => (
            <section key={category}>
              <h3>
                {memoryCategoryLabels[category] || category.replaceAll("_", " ")}
              </h3>
              {items.map((record) => (
                <article className="memory-record" key={record.id}>
                  <p>{record.memory}</p>
                  <div>
                    <span>
                      {formatTime(record.updated_at || record.created_at)}
                      {record.metadata.active_category
                        ? ` · ${productCategoryLabels[record.metadata.active_category] || record.metadata.active_category}`
                        : ""}
                    </span>
                    <button type="button" onClick={() => onDelete(record.id)}>
                      Quên
                    </button>
                  </div>
                </article>
              ))}
            </section>
          ))}
        </div>
        {records.length > 0 ? (
          <button className="forget-all-button" type="button" onClick={onDeleteAll}>
            Quên toàn bộ
          </button>
        ) : null}
      </aside>
    </div>
  );
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, emptyState);
  const [message, setMessage] = useState("");
  const [session, setSession] = useState<Session | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [loginOpen, setLoginOpen] = useState(false);
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false);
  const [memoryRecords, setMemoryRecords] = useState<MemoryRecord[]>([]);
  const [memoryCount, setMemoryCount] = useState(0);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [memoryNotice, setMemoryNotice] = useState<{
    status: "updating" | "success" | "failed";
    message: string;
  } | null>(null);
  const requestInFlight = useRef(false);
  const activeRequest = useRef<AbortController | null>(null);
  const memoryPoll = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const authInitialized = useRef(false);
  const identityNamespace = useRef<string | null>(null);
  const restoreStarted = useRef<string | null>(null);
  const userId = session?.user.id || null;
  const accessToken = session?.access_token || null;
  const storageKey = storageKeyFor(userId);

  const refreshMemories = useCallback(
    async (signal?: AbortSignal) => {
      if (!accessToken) {
        setMemoryRecords([]);
        setMemoryCount(0);
        return;
      }
      setMemoryLoading(true);
      setMemoryError(null);
      try {
        const result = await getMemories(accessToken, signal);
        setMemoryRecords(result.results);
        setMemoryCount(result.count);
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") return;
        setMemoryError(messageFromError(error));
      } finally {
        setMemoryLoading(false);
      }
    },
    [accessToken],
  );

  const beginMemoryWritePolling = useCallback(
    (threadId: string, write: MemoryWriteSummary | undefined, token: string | null) => {
      memoryPoll.current?.abort();
      if (!write || write.status === "skipped") return;
      if (write.status === "failed" || !token) {
        setMemoryNotice({
          status: "failed",
          message: "Hồ sơ tạm thời chưa được cập nhật.",
        });
        return;
      }
      const controller = new AbortController();
      memoryPoll.current = controller;
      setMemoryNotice({ status: "updating", message: "Đang cập nhật hồ sơ đã nhớ…" });
      void (async () => {
        try {
          for (let attempt = 0; attempt < 20; attempt += 1) {
            const result = await getMemoryWriteStatus(
              threadId,
              token,
              controller.signal,
            );
            if (result.status === "succeeded") {
              if (result.added_count > 0) {
                setMemoryNotice({
                  status: "success",
                  message: `Đã ghi nhớ thêm ${result.added_count} thông tin`,
                });
                try {
                  const memories = await getMemories(token, controller.signal);
                  setMemoryRecords(memories.results);
                  setMemoryCount(memories.count);
                } catch {
                  // The write succeeded; profile refresh can be retried from the panel.
                }
              } else {
                setMemoryNotice({
                  status: "success",
                  message: "Không có thông tin mới cần ghi nhớ",
                });
              }
              return;
            }
            if (result.status === "failed") {
              setMemoryNotice({
                status: "failed",
                message: "Hồ sơ tạm thời chưa được cập nhật.",
              });
              return;
            }
            await wait(750, controller.signal);
          }
          setMemoryNotice({
            status: "failed",
            message: "Hồ sơ đang xử lý lâu hơn dự kiến; bạn vẫn có thể tiếp tục.",
          });
        } catch (error) {
          if (error instanceof Error && error.name === "AbortError") return;
          setMemoryNotice({
            status: "failed",
            message: "Hồ sơ tạm thời chưa được cập nhật.",
          });
        } finally {
          if (memoryPoll.current === controller) memoryPoll.current = null;
        }
      })();
    },
    [],
  );

  useEffect(() => {
    let alive = true;
    const applySession = (nextSession: Session | null) => {
      if (!alive) return;
      const nextNamespace = nextSession?.user.id || "anonymous";
      if (!authInitialized.current) {
        authInitialized.current = true;
        identityNamespace.current = nextNamespace;
        setSession(nextSession);
        dispatch({
          type: "LOAD_STATE",
          state: loadState(storageKeyFor(nextSession?.user.id || null)),
        });
        setAuthReady(true);
        return;
      }
      if (identityNamespace.current !== nextNamespace) {
        activeRequest.current?.abort();
        memoryPoll.current?.abort();
        requestInFlight.current = false;
        restoreStarted.current = null;
        setMessage("");
        setMemoryNotice(null);
        setMemoryRecords([]);
        setMemoryCount(0);
        setMemoryPanelOpen(false);
        setLoginOpen(false);
        dispatch({ type: "RESET" });
      }
      identityNamespace.current = nextNamespace;
      setSession(nextSession);
      setAuthReady(true);
    };

    if (!supabase) {
      applySession(null);
      return () => {
        alive = false;
      };
    }
    void supabase.auth
      .getSession()
      .then(({ data }) => applySession(data.session))
      .catch(() => applySession(null));
    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) =>
      applySession(nextSession),
    );
    return () => {
      alive = false;
      data.subscription.unsubscribe();
    };
  }, []);

  const restoreThread = useCallback(
    async (
      threadId: string,
      token: string | null,
      signal: AbortSignal,
    ) => {
      dispatch({
        type: "SET_PHASE",
        phase: "restoring",
        progress: "Đang khôi phục cuộc trò chuyện…",
      });
      try {
        for (let attempt = 0; attempt < 30; attempt += 1) {
          const status = await getThreadStatus(threadId, token, signal);
          if (
            status.status === "waiting_for_memory_confirmation" &&
            status.memory_confirmation
          ) {
            dispatch({
              type: "RECONCILE_MEMORY",
              confirmation: status.memory_confirmation,
            });
            return;
          }
          if (status.status === "waiting_for_clarification") {
            dispatch({
              type: "RECONCILE_WAITING",
              questions: status.questions,
            });
            return;
          }
          if (status.status === "completed") {
            dispatch({
              type: "RECONCILE_COMPLETED",
              answer: status.answer || "",
              products: status.selected_products || [],
            });
            beginMemoryWritePolling(threadId, status.memory_write || undefined, token);
            return;
          }
          dispatch({
            type: "SET_PROGRESS",
            progress: "Cuộc trò chuyện vẫn đang được xử lý…",
          });
          await wait(2_000, signal);
        }
        dispatch({
          type: "REQUEST_FAILED",
          message:
            "Server vẫn đang xử lý lâu hơn dự kiến. Bạn có thể kiểm tra lại trạng thái.",
        });
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") return;
        if (error instanceof ApiError && error.status === 404) {
          localStorage.removeItem(storageKeyFor(token ? userId : null));
          dispatch({
            type: "RESET",
            message: "Phiên đã lưu không còn tồn tại trên server.",
          });
          return;
        }
        dispatch({ type: "REQUEST_FAILED", message: messageFromError(error) });
      }
    },
    [beginMemoryWritePolling, userId],
  );

  useEffect(() => {
    if (!authReady || state.phase !== "restoring" || !state.threadId) return;
    const restoreKey = `${storageKey}:${state.threadId}`;
    if (restoreStarted.current === restoreKey) return;
    restoreStarted.current = restoreKey;
    const controller = new AbortController();
    void restoreThread(state.threadId, accessToken, controller.signal);
    return () => controller.abort();
  }, [accessToken, authReady, restoreThread, state.phase, state.threadId, storageKey]);

  useEffect(() => {
    if (!authReady) return;
    if (!state.threadId) {
      localStorage.removeItem(storageKey);
      return;
    }
    try {
      localStorage.setItem(
        storageKey,
        JSON.stringify({
          version: 2,
          threadId: state.threadId,
          items: state.items,
        }),
      );
    } catch {
      // The live conversation remains usable when browser storage is unavailable.
    }
  }, [authReady, state.threadId, state.items, storageKey]);

  useEffect(() => {
    if (!authReady || !accessToken) return;
    const controller = new AbortController();
    void refreshMemories(controller.signal);
    return () => controller.abort();
  }, [accessToken, authReady, refreshMemories]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [state.items, state.progress, state.error]);

  useEffect(
    () => () => {
      activeRequest.current?.abort();
      memoryPoll.current?.abort();
    },
    [],
  );

  const activeClarification = useMemo(
    () =>
      state.items.findLast(
        (item): item is ClarificationItem =>
          item.type === "clarification" && !item.submitted,
      ),
    [state.items],
  );
  const activeMemoryConfirmation = useMemo(
    () =>
      state.items.findLast(
        (item): item is MemoryConfirmationItem =>
          item.type === "memory_confirmation" && !item.submitted,
      ),
    [state.items],
  );
  const busy =
    state.phase === "sending" ||
    state.phase === "resuming" ||
    state.phase === "restoring";
  const canSend =
    authReady &&
    state.phase === "idle" &&
    !activeClarification &&
    !activeMemoryConfirmation;

  const handleStreamEvent = (
    event: SseEvent,
    assistantId: string,
    markTerminal: () => void,
    rememberThread: (threadId: string) => void,
  ) => {
    switch (event.event) {
      case "session":
        rememberThread(event.data.thread_id);
        dispatch({ type: "SET_THREAD", threadId: event.data.thread_id });
        break;
      case "progress":
        dispatch({
          type: "SET_PROGRESS",
          progress: progressLabels[event.data.stage] || "Đang xử lý…",
        });
        break;
      case "token":
        dispatch({
          type: "APPEND_TOKEN",
          id: assistantId,
          delta: event.data.delta,
        });
        break;
      case "clarification_required":
        markTerminal();
        rememberThread(event.data.thread_id);
        dispatch({ type: "SET_THREAD", threadId: event.data.thread_id });
        dispatch({
          type: "ADD_CLARIFICATION",
          assistantId,
          questions: event.data.questions,
        });
        break;
      case "memory_confirmation_required":
        markTerminal();
        rememberThread(event.data.thread_id);
        dispatch({ type: "SET_THREAD", threadId: event.data.thread_id });
        dispatch({
          type: "ADD_MEMORY_CONFIRMATION",
          assistantId,
          confirmation: {
            message: event.data.message,
            category: event.data.category,
            candidates: event.data.candidates,
          },
        });
        break;
      case "completed":
        markTerminal();
        rememberThread(event.data.thread_id);
        dispatch({ type: "SET_THREAD", threadId: event.data.thread_id });
        dispatch({
          type: "COMPLETE",
          id: assistantId,
          answer: event.data.answer,
          products: event.data.selected_products || [],
        });
        beginMemoryWritePolling(
          event.data.thread_id,
          event.data.memory_write,
          accessToken,
        );
        break;
      case "error":
        markTerminal();
        throw new Error(event.data.message);
    }
  };

  const sendMessage = async (event?: FormEvent, suggestedMessage?: string) => {
    event?.preventDefault();
    const trimmed = (suggestedMessage ?? message).trim();
    if (!trimmed || !canSend || requestInFlight.current) return;

    requestInFlight.current = true;
    const controller = new AbortController();
    activeRequest.current = controller;
    const assistantId = createId();
    let activeThreadId = state.threadId;
    let terminal = false;
    dispatch({
      type: "ADD_USER",
      item: { id: createId(), type: "user", text: trimmed },
    });
    dispatch({
      type: "ADD_ASSISTANT",
      item: {
        id: assistantId,
        type: "assistant",
        text: "",
        products: [],
        streaming: true,
      },
    });
    dispatch({
      type: "SET_PHASE",
      phase: "sending",
      progress: "Đang hiểu nhu cầu…",
    });
    setMessage("");

    try {
      await streamChat(
        trimmed,
        state.threadId,
        (sseEvent) =>
          handleStreamEvent(
            sseEvent,
            assistantId,
            () => {
              terminal = true;
            },
            (threadId) => {
              activeThreadId = threadId;
            },
          ),
        accessToken,
        controller.signal,
      );
      if (!terminal)
        throw new Error("Stream kết thúc trước khi có kết quả cuối.");
    } catch (error) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        dispatch({
          type: "REQUEST_FAILED",
          assistantId,
          message: messageFromError(error),
        });
        if (activeThreadId)
          dispatch({ type: "SET_THREAD", threadId: activeThreadId });
      }
    } finally {
      requestInFlight.current = false;
      activeRequest.current = null;
    }
  };

  const submitClarification = async (
    item: ClarificationItem,
    answers: ClarificationAnswer[],
  ) => {
    if (requestInFlight.current || !state.threadId) return;
    if (answers.length !== item.questions.length) return;

    requestInFlight.current = true;
    const controller = new AbortController();
    activeRequest.current = controller;
    const assistantId = createId();
    let terminal = false;
    dispatch({ type: "MARK_SUBMITTED", itemId: item.id });
    dispatch({
      type: "ADD_ASSISTANT",
      item: {
        id: assistantId,
        type: "assistant",
        text: "",
        products: [],
        streaming: true,
      },
    });
    dispatch({
      type: "SET_PHASE",
      phase: "resuming",
      progress: "Đang tìm sản phẩm…",
    });

    try {
      await resumeChat(
        state.threadId,
        answers.map((answer) => ({
          question_id: answer.question_id,
          option_id: answer.option_id,
          ...(answer.option_id === "other"
            ? { custom_answer: answer.custom_answer?.trim() }
            : {}),
        })),
        (sseEvent) =>
          handleStreamEvent(
            sseEvent,
            assistantId,
            () => {
              terminal = true;
            },
            () => undefined,
          ),
        accessToken,
        controller.signal,
      );
      if (!terminal)
        throw new Error("Stream kết thúc trước khi có kết quả cuối.");
    } catch (error) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        dispatch({
          type: "REQUEST_FAILED",
          assistantId,
          message: messageFromError(error),
        });
      }
    } finally {
      requestInFlight.current = false;
      activeRequest.current = null;
    }
  };

  const submitMemoryConfirmation = async (
    item: MemoryConfirmationItem,
    decisions: MemoryConfirmationDecision[],
  ) => {
    if (requestInFlight.current || !state.threadId || !accessToken) return;
    if (decisions.length !== item.confirmation.candidates.length) return;
    requestInFlight.current = true;
    const controller = new AbortController();
    activeRequest.current = controller;
    const assistantId = createId();
    let terminal = false;
    dispatch({ type: "MARK_MEMORY_SUBMITTED", itemId: item.id });
    dispatch({
      type: "ADD_ASSISTANT",
      item: {
        id: assistantId,
        type: "assistant",
        text: "",
        products: [],
        streaming: true,
      },
    });
    dispatch({
      type: "SET_PHASE",
      phase: "resuming",
      progress: "Đang áp dụng thông tin đã xác nhận…",
    });
    try {
      await resumeMemoryConfirmation(
        state.threadId,
        decisions.map((decision) => ({
          candidate_id: decision.candidate_id,
          action: decision.action,
          ...(decision.action === "edit"
            ? {
                option_id: decision.option_id,
                ...(decision.option_id === "other"
                  ? { custom_answer: decision.custom_answer?.trim() }
                  : {}),
              }
            : {}),
        })),
        (sseEvent) =>
          handleStreamEvent(
            sseEvent,
            assistantId,
            () => {
              terminal = true;
            },
            () => undefined,
          ),
        accessToken,
        controller.signal,
      );
      if (!terminal)
        throw new Error("Stream kết thúc trước khi có kết quả cuối.");
    } catch (error) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        dispatch({
          type: "REQUEST_FAILED",
          assistantId,
          message: messageFromError(error),
        });
      }
    } finally {
      requestInFlight.current = false;
      activeRequest.current = null;
    }
  };

  const checkStatus = () => {
    if (!state.threadId || requestInFlight.current) return;
    requestInFlight.current = true;
    const controller = new AbortController();
    activeRequest.current = controller;
    void restoreThread(state.threadId, accessToken, controller.signal).finally(() => {
      requestInFlight.current = false;
      activeRequest.current = null;
    });
  };

  const startNewConversation = () => {
    if (busy || memoryNotice?.status === "updating") return;
    localStorage.removeItem(storageKey);
    setMessage("");
    dispatch({ type: "RESET" });
  };

  const signIn = async (event: FormEvent) => {
    event.preventDefault();
    if (!supabase || authBusy || !loginEmail.trim() || !loginPassword) return;
    setAuthBusy(true);
    setAuthError(null);
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email: loginEmail.trim(),
        password: loginPassword,
      });
      if (error) setAuthError("Email hoặc mật khẩu không đúng.");
      else {
        setLoginPassword("");
        setLoginOpen(false);
      }
    } catch {
      setAuthError("Không thể kết nối tới dịch vụ đăng nhập.");
    } finally {
      setAuthBusy(false);
    }
  };

  const signOut = async () => {
    if (!supabase || authBusy) return;
    setAuthBusy(true);
    setAuthError(null);
    try {
      const { error } = await supabase.auth.signOut();
      if (error) setAuthError("Chưa thể đăng xuất. Vui lòng thử lại.");
    } catch {
      setAuthError("Không thể kết nối tới dịch vụ đăng nhập.");
    } finally {
      setAuthBusy(false);
    }
  };

  const handleDeleteMemory = async (memoryId: string) => {
    if (!accessToken) return;
    try {
      await deleteMemory(memoryId, accessToken);
      await refreshMemories();
    } catch (error) {
      setMemoryError(messageFromError(error));
    }
  };

  const handleDeleteAllMemories = async () => {
    if (!accessToken) return;
    if (!window.confirm("Quên toàn bộ hồ sơ dài hạn? Thao tác này không thể hoàn tác."))
      return;
    try {
      await deleteAllMemories(accessToken);
      setMemoryRecords([]);
      setMemoryCount(0);
    } catch (error) {
      setMemoryError(messageFromError(error));
    }
  };

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  const composerPlaceholder = activeClarification || activeMemoryConfirmation
    ? "Hãy hoàn tất các lựa chọn phía trên"
    : busy
      ? "Product Advisor đang xử lý…"
      : state.phase === "error"
        ? "Kiểm tra trạng thái trước khi gửi tiếp"
        : "Nhập nhu cầu của bạn…";
  const messageWordCount = countWords(message);
  const lastAssistantIndex = state.items.findLastIndex(
    (item) => item.type === "assistant" && !item.streaming,
  );

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <img className="brand-mark" src="/advisor-logo.png" alt="" />
          <div>
            <strong>Trợ lý AI mua sắm</strong>
            <span>Tư vấn sản phẩm theo nhu cầu của bạn</span>
          </div>
        </div>
        <div className="header-actions">
          <div className="online-status">
            <span aria-hidden="true" /> Trực tuyến
          </div>
          {session ? (
            <>
              <button
                className="memory-profile-button"
                type="button"
                onClick={() => setMemoryPanelOpen(true)}
              >
                {memoryNotice?.status === "updating"
                  ? "Đang cập nhật"
                  : "Hồ sơ đang nhớ"}{" "}
                <strong>{memoryCount}</strong>
              </button>
              <div className="signed-in-user" title={session.user.email || undefined}>
                <span>Cá nhân hóa đang bật</span>
                <strong>{session.user.email || "Tài khoản demo"}</strong>
              </div>
              <button
                className="auth-button"
                type="button"
                disabled={authBusy}
                onClick={() => void signOut()}
              >
                Đăng xuất
              </button>
            </>
          ) : authConfigured ? (
            <button
              className="auth-button"
              type="button"
              disabled={!authReady || authBusy}
              onClick={() => {
                setAuthError(null);
                setLoginOpen(true);
              }}
            >
              Đăng nhập
            </button>
          ) : (
            <span className="anonymous-status">Ẩn danh</span>
          )}
          <button
            className="new-chat-button"
            type="button"
            disabled={
              busy ||
              memoryNotice?.status === "updating" ||
              (!state.threadId && state.items.length === 0)
            }
            onClick={startNewConversation}
          >
            <span aria-hidden="true">↻</span> Cuộc trò chuyện mới
          </button>
        </div>
      </header>

      {loginOpen && !session ? (
        <div
          className="login-backdrop"
          role="presentation"
          onMouseDown={() => !authBusy && setLoginOpen(false)}
        >
          <form
            className="login-card"
            onSubmit={(event) => void signIn(event)}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button
              className="login-close"
              type="button"
              aria-label="Đóng đăng nhập"
              disabled={authBusy}
              onClick={() => setLoginOpen(false)}
            >
              ×
            </button>
            <span>Tài khoản demo</span>
            <h2>Đăng nhập để bật cá nhân hóa</h2>
            <p>Hồ sơ được dùng xuyên các cuộc trò chuyện của riêng tài khoản này.</p>
            <label>
              Email
              <input
                type="email"
                autoComplete="email"
                required
                value={loginEmail}
                disabled={authBusy}
                onChange={(event) => setLoginEmail(event.target.value)}
              />
            </label>
            <label>
              Mật khẩu
              <input
                type="password"
                autoComplete="current-password"
                required
                value={loginPassword}
                disabled={authBusy}
                onChange={(event) => setLoginPassword(event.target.value)}
              />
            </label>
            {authError ? <div className="login-error" role="alert">{authError}</div> : null}
            <button className="primary-button" type="submit" disabled={authBusy}>
              {authBusy ? "Đang đăng nhập…" : "Đăng nhập"}
            </button>
          </form>
        </div>
      ) : null}

      <main className="chat-main">
        <div className="conversation" aria-live="polite">
          {state.items.length === 0 ? (
            <>
              <section className="welcome-card">
                <div className="welcome-icon">
                  <img src="/advisor-logo.png" alt="Trợ lý AI" />
                </div>
                <h1>Bạn đang tìm sản phẩm nào?</h1>
                <p>
                  Hãy mô tả nhu cầu, ngân sách hoặc điều bạn quan tâm.
                  <br />
                  <strong>Trợ lý AI</strong> sẽ giúp bạn thu hẹp lựa chọn.
                </p>
                <div
                  className="suggestion-list starter-suggestions"
                  aria-label="Gợi ý câu hỏi"
                >
                  {starterSuggestions.map((suggestion) => (
                    <button
                      type="button"
                      key={suggestion}
                      disabled={!canSend}
                      onClick={() => void sendMessage(undefined, suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </section>
              <div className="message-row assistant-row greeting-row">
                <img
                  className="assistant-avatar"
                  src="/advisor-logo.png"
                  alt=""
                />
                <div className="assistant-result">
                  <div className="assistant-name">Trợ lý AI</div>
                  <div className="message assistant-message greeting-message">
                    Xin chào! Mình có thể giúp bạn tìm sản phẩm phù hợp với nhu
                    cầu và ngân sách. Bạn đang quan tâm đến sản phẩm nào ạ?
                  </div>
                </div>
              </div>
            </>
          ) : null}

          {state.items.map((item, itemIndex) => {
            if (item.type === "user") {
              return (
                <div className="message-row user-row" key={item.id}>
                  <div className="message user-message">{item.text}</div>
                </div>
              );
            }
            if (item.type === "assistant") {
              if (!item.text && item.products.length === 0) return null;
              return (
                <div className="message-row assistant-row" key={item.id}>
                  <img
                    className="assistant-avatar"
                    src="/advisor-logo.png"
                    alt=""
                  />
                  <div className="assistant-result">
                    <div className="assistant-name">Trợ lý AI</div>
                    {item.text ? (
                      <div className="message assistant-message">
                        <MarkdownText text={item.text} />
                      </div>
                    ) : null}
                    {item.products.length > 0 ? (
                      <div className="product-list">
                        {item.products.slice(0, 3).map((product) => (
                          <ProductCard
                            key={product.product_id}
                            product={product}
                          />
                        ))}
                      </div>
                    ) : null}
                    {itemIndex === lastAssistantIndex && canSend ? (
                      <div className="follow-up-block">
                        <span>Bạn có thể hỏi tiếp</span>
                        <div className="suggestion-list follow-up-suggestions">
                          {followUpSuggestions.map((suggestion) => (
                            <button
                              type="button"
                              key={suggestion}
                              onClick={() =>
                                void sendMessage(undefined, suggestion)
                              }
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              );
            }
            if (item.type === "memory_confirmation") {
              return (
                <div className="message-row assistant-row" key={item.id}>
                  <img
                    className="assistant-avatar"
                    src="/advisor-logo.png"
                    alt=""
                  />
                  <MemoryConfirmationCard
                    item={item}
                    disabled={busy || item.submitted}
                    onSubmit={(decisions) =>
                      void submitMemoryConfirmation(item, decisions)
                    }
                  />
                </div>
              );
            }
            return (
              <div className="message-row assistant-row" key={item.id}>
                <img
                  className="assistant-avatar"
                  src="/advisor-logo.png"
                  alt=""
                />
                <ClarificationCard
                  item={item}
                  disabled={busy || item.submitted}
                  onSelect={(questionId, optionId) =>
                    dispatch({
                      type: "SELECT_OPTION",
                      itemId: item.id,
                      questionId,
                      optionId,
                    })
                  }
                  onCustomAnswer={(questionId, value) =>
                    dispatch({
                      type: "SET_CUSTOM_ANSWER",
                      itemId: item.id,
                      questionId,
                      value,
                    })
                  }
                  onConfirm={(questionId) =>
                    dispatch({
                      type: "CONFIRM_ANSWER",
                      itemId: item.id,
                      questionId,
                    })
                  }
                  onSubmit={(answers) =>
                    void submitClarification(item, answers)
                  }
                />
              </div>
            );
          })}

          {state.progress ? (
            <div className="status-row" role="status">
              <span className="status-dot" aria-hidden="true" />
              {state.progress}
            </div>
          ) : null}

          {memoryNotice ? (
            <div className={`memory-notice ${memoryNotice.status}`} role="status">
              <span aria-hidden="true">
                {memoryNotice.status === "updating"
                  ? "↻"
                  : memoryNotice.status === "success"
                    ? "✓"
                    : "!"}
              </span>
              {memoryNotice.message}
            </div>
          ) : null}

          {authError && !loginOpen ? (
            <div className="error-banner" role="alert">
              <div>
                <strong>Lỗi tài khoản</strong>
                <span>{authError}</span>
              </div>
              <button type="button" onClick={() => setAuthError(null)}>Đóng</button>
            </div>
          ) : null}

          {state.error ? (
            <div className="error-banner" role="alert">
              <div>
                <strong>Chưa thể hoàn tất yêu cầu</strong>
                <span>{state.error}</span>
              </div>
              <button
                type="button"
                onClick={
                  state.threadId
                    ? checkStatus
                    : () => dispatch({ type: "CLEAR_ERROR" })
                }
              >
                {state.threadId ? "Kiểm tra trạng thái" : "Đóng"}
              </button>
            </div>
          ) : null}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="composer-shell">
        <form
          className="composer"
          onSubmit={(event) => void sendMessage(event)}
        >
          <textarea
            value={message}
            onChange={(event) =>
              setMessage(limitWords(event.target.value, MAX_MESSAGE_WORDS))
            }
            onKeyDown={onComposerKeyDown}
            placeholder={composerPlaceholder}
            rows={2}
            disabled={!canSend}
            aria-label="Tin nhắn"
          />
          <button
            className="send-button"
            type="submit"
            disabled={!canSend || !message.trim()}
            aria-label="Gửi tin nhắn"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="m3 11 18-8-8 18-2-8-8-2Z" />
              <path d="m11 13 4-4" />
            </svg>
            Gửi
          </button>
        </form>
        <div className="composer-meta">
          <span>
            Giá, tồn kho và khuyến mãi có thể thay đổi, cần xác nhận lại trước
            khi mua.
          </span>
          <strong
            className={messageWordCount >= MAX_MESSAGE_WORDS ? "at-limit" : ""}
          >
            {messageWordCount}/{MAX_MESSAGE_WORDS} từ
          </strong>
        </div>
      </footer>

      {memoryPanelOpen && session ? (
        <MemoryPanel
          records={memoryRecords}
          count={memoryCount}
          loading={memoryLoading}
          error={memoryError}
          onClose={() => setMemoryPanelOpen(false)}
          onRefresh={() => void refreshMemories()}
          onDelete={(id) => void handleDeleteMemory(id)}
          onDeleteAll={() => void handleDeleteAllMemories()}
        />
      ) : null}
    </div>
  );
}
