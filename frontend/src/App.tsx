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
  SelectedProduct,
  SseEvent,
  SuggestionConversationMessage,
  getThreadStatus,
  resolveProductImageUrl,
  resumeChat,
  streamChat,
  streamSuggestions,
} from "./api";

const STORAGE_KEY = "product-advisor:mvp:v1";

type UserItem = { id: string; type: "user"; text: string };
type AssistantItem = {
  id: string;
  type: "assistant";
  text: string;
  products: SelectedProduct[];
  streaming?: boolean;
  suggestions?: string[];
  suggestionsRequested?: boolean;
};
type ClarificationItem = {
  id: string;
  type: "clarification";
  questions: ClarificationQuestion[];
  answers: Record<string, ClarificationAnswer>;
  confirmedIds: string[];
  submitted: boolean;
};
type TimelineItem = UserItem | AssistantItem | ClarificationItem;
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
  | { type: "RECONCILE_WAITING"; questions: ClarificationQuestion[] }
  | { type: "RECONCILE_COMPLETED"; answer: string; products: SelectedProduct[] }
  | { type: "REQUEST_SUGGESTIONS"; id: string }
  | { type: "SET_SUGGESTIONS"; id: string; suggestions: string[] }
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

function loadState(): ChatState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyState;
    const saved = JSON.parse(raw) as {
      version?: number;
      threadId?: unknown;
      items?: unknown;
    };
    if (
      saved.version !== 1 ||
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
    localStorage.removeItem(STORAGE_KEY);
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
        const currentAssistant = items[lastAssistant] as AssistantItem;
        items[lastAssistant] = {
          ...currentAssistant,
          text: action.answer,
          products,
          streaming: false,
          suggestionsRequested: Boolean(currentAssistant.suggestions?.length),
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
    case "REQUEST_SUGGESTIONS":
      return {
        ...state,
        items: state.items.map((item) =>
          item.id === action.id && item.type === "assistant"
            ? { ...item, suggestionsRequested: true }
            : item,
        ),
      };
    case "SET_SUGGESTIONS":
      return {
        ...state,
        items: state.items.map((item) =>
          item.id === action.id && item.type === "assistant"
            ? { ...item, suggestions: action.suggestions.slice(0, 4) }
            : item,
        ),
      };
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
  need_extracted: "Đang hiểu nhu cầu…",
  clarification_ready: "Đang chuẩn bị câu hỏi…",
  clarification_completed: "Đang tìm sản phẩm…",
  filter_built: "Đang tìm sản phẩm…",
  retrieval_completed: "Đang tìm sản phẩm…",
  ranking_completed: "Đang hoàn thiện gợi ý…",
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

function suggestionConversation(
  items: TimelineItem[],
): SuggestionConversationMessage[] {
  const messages: SuggestionConversationMessage[] = [];
  for (const item of items) {
    if (item.type === "user") {
      messages.push({ role: "user", content: item.text });
      continue;
    }
    if (item.type === "assistant") {
      if (item.text.trim())
        messages.push({ role: "assistant", content: item.text });
      continue;
    }
    const questionText = item.questions
      .map((question) => question.question)
      .join(" ");
    if (questionText)
      messages.push({ role: "assistant", content: questionText });
    const answerText = item.questions
      .map((question) => {
        const answer = item.answers[question.question_id];
        if (!answer) return null;
        const value =
          answer.option_id === "other"
            ? answer.custom_answer?.trim()
            : question.options.find(
                (option) => option.option_id === answer.option_id,
              )?.label;
        return value ? `${question.question}: ${value}` : null;
      })
      .filter((value): value is string => Boolean(value))
      .join("; ");
    if (answerText)
      messages.push({
        role: "user",
        content: `Thông tin bổ sung: ${answerText}`,
      });
  }

  const selected: SuggestionConversationMessage[] = [];
  let characters = 0;
  for (const candidate of messages.slice(-40).reverse()) {
    if (characters + candidate.content.length > 40_000) break;
    selected.push(candidate);
    characters += candidate.content.length;
  }
  return selected.reverse();
}
function limitWords(value: string, maximum: number): string {
  const matches = [...value.matchAll(/\S+/g)];
  if (matches.length <= maximum) return value;
  const lastWord = matches[maximum - 1];
  return value.slice(0, (lastWord.index ?? 0) + lastWord[0].length);
}

function messageFromError(error: unknown): string {
  if (error instanceof ApiError) {
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

function displayProductName(product: SelectedProduct): string {
  const name = product.name?.trim();
  if (!name) return product.product_id;
  return name.charAt(0).toLocaleUpperCase("vi-VN") + name.slice(1);
}

function productImageSources(product: SelectedProduct): {
  primary: string | null;
  fallback: string | null;
} {
  const raw = (product.image_url || product.image_path)?.trim();
  if (!raw) return { primary: null, fallback: null };

  const primary = resolveProductImageUrl(raw);
  const filename = raw.split(/[\\/]/).filter(Boolean).pop();
  const fallback =
    /^(?:https?:|data:|blob:)/i.test(raw) || !filename
      ? null
      : `/${encodeURIComponent(filename)}`;
  return { primary, fallback };
}

function MarkdownText({ text }: { text: string }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

function AiFeedbackCard({ product }: { product: SelectedProduct }) {
  const [feedback, setFeedback] = useState("");
  const [rating, setRating] = useState(0);
  const [hoveredRating, setHoveredRating] = useState(0);
  const [submitted, setSubmitted] = useState(false);
  const productImage = productImageSources(product);

  const submitFeedback = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitted(true);
  };

  return (
    <div className="ai-feedback-block">
      <a
        className="mock-product-link"
        href="#"
        onClick={(event) => event.preventDefault()}
        aria-label={`Link minh họa đến trang ${displayProductName(product)}`}
      >
        {productImage.primary ? (
          <img
            className="mock-link-image"
            src={productImage.primary}
            alt=""
            onError={(event) => {
              if (
                productImage.fallback &&
                event.currentTarget.src !==
                  new URL(productImage.fallback, window.location.href).href
              ) {
                event.currentTarget.src = productImage.fallback;
              } else {
                event.currentTarget.hidden = true;
              }
            }}
          />
        ) : (
          <span className="mock-link-icon" aria-hidden="true">
            ↗
          </span>
        )}
        <span>
          <small>Trang sản phẩm</small>
          <strong>{displayProductName(product)}</strong>
        </span>
        <span className="mock-link-label">Link minh họa</span>
      </a>

      <section className="ai-feedback-card" aria-labelledby="ai-feedback-title">
        {submitted ? (
          <div className="feedback-success" role="status">
            <span aria-hidden="true">✓</span>
            <div>
              <h3>Cảm ơn phản hồi của bạn!</h3>
              <p>Phản hồi đang được ghi nhận ở giao diện thử nghiệm.</p>
            </div>
          </div>
        ) : (
          <form className="feedback-form" onSubmit={submitFeedback}>
            <div className="feedback-heading">
              <span className="feedback-ai-icon" aria-hidden="true">
                ✦
              </span>
              <div>
                <h3 id="ai-feedback-title">
                  Bạn hài lòng với tư vấn của AI không?
                </h3>
                <p>Phản hồi của bạn giúp trợ lý tư vấn tốt hơn.</p>
              </div>
            </div>

            <fieldset className="rating-fieldset">
              <legend>Mức độ hài lòng</legend>
              <div
                className="star-rating"
                onMouseLeave={() => setHoveredRating(0)}
              >
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    type="button"
                    className={
                      star <= (hoveredRating || rating) ? "active" : ""
                    }
                    onMouseEnter={() => setHoveredRating(star)}
                    onFocus={() => setHoveredRating(star)}
                    onBlur={() => setHoveredRating(0)}
                    onClick={() => setRating(star)}
                    aria-label={`${star} sao`}
                    aria-pressed={rating === star}
                  >
                    ★
                  </button>
                ))}
                <span>{rating ? `${rating}/5 sao` : "Chọn số sao"}</span>
              </div>
            </fieldset>

            <label className="feedback-comment">
              Chia sẻ thêm <span>(không bắt buộc)</span>
              <textarea
                value={feedback}
                onChange={(event) => setFeedback(event.target.value)}
                placeholder="AI đã hỗ trợ tốt điều gì hoặc cần cải thiện điều gì?"
                rows={3}
              />
            </label>

            <button
              className="feedback-submit"
              type="submit"
              disabled={rating === 0}
            >
              Gửi phản hồi
            </button>
          </form>
        )}
      </section>
    </div>
  );
}

function ProductCard({
  product,
  onOpen,
}: {
  product: SelectedProduct;
  onOpen: () => void;
}) {
  const promotional = finitePrice(product.promotional_price_vnd)
    ? product.promotional_price_vnd
    : null;
  const original = finitePrice(product.original_price_vnd)
    ? product.original_price_vnd
    : null;
  const effective = finitePrice(product.effective_price_vnd)
    ? product.effective_price_vnd
    : null;
  // const currentPrice = promotional ?? effective ?? original;
  const productImage = productImageSources(product);

  const validOriginal = original !== -1 ? original : null;
  const validPromotional = promotional !== -1 ? promotional : null;

  const currentPrice = validPromotional ?? validOriginal;

  return (
    <article
      className="product-card"
      role="button"
      tabIndex={0}
      aria-label={`Mở trang ${displayProductName(product)}`}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen();
        }
      }}
    >
      {productImage.primary ? (
        <img
          className="product-image"
          src={productImage.primary}
          alt={displayProductName(product)}
          onError={(event) => {
            if (
              productImage.fallback &&
              event.currentTarget.src !==
                new URL(productImage.fallback, window.location.href).href
            ) {
              event.currentTarget.src = productImage.fallback;
            } else {
              event.currentTarget.hidden = true;
            }
          }}
        />
      ) : null}
      <div className="product-content">
        <div className="product-heading">
          <h3>{displayProductName(product)}</h3>
          {product.name ? (
            <span className="product-id">{product.product_id}</span>
          ) : null}
        </div>
        {currentPrice !== null ? (
          <div className="product-price">
            <strong>{formatPrice(currentPrice)}</strong>
            {validOriginal !== null &&
            validPromotional !== null &&
            validOriginal > validPromotional ? (
              <del>{formatPrice(validOriginal)}</del>
            ) : null}
          </div>
        ) : (
          <div className="product-price">
            <strong>Liên hệ với nhân viên</strong>
          </div>
        )}
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
        <div className="product-review-hint">
          <span aria-hidden="true">↗</span>
          Xem trang sản phẩm
          <span className="product-review-arrow" aria-hidden="true">
            →
          </span>
        </div>
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

export default function App() {
  const [state, dispatch] = useReducer(reducer, undefined, loadState);
  const [message, setMessage] = useState("");
  const [feedbackTarget, setFeedbackTarget] = useState<{
    assistantId: string;
    product: SelectedProduct;
  } | null>(null);
  const requestInFlight = useRef(false);
  const activeRequest = useRef<AbortController | null>(null);
  const suggestionRequest = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const initialThreadId = useRef(state.threadId);

  const restoreThread = useCallback(
    async (threadId: string, signal: AbortSignal) => {
      dispatch({
        type: "SET_PHASE",
        phase: "restoring",
        progress: "Đang khôi phục cuộc trò chuyện…",
      });
      try {
        for (let attempt = 0; attempt < 30; attempt += 1) {
          const status = await getThreadStatus(threadId, signal);
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
          localStorage.removeItem(STORAGE_KEY);
          dispatch({
            type: "RESET",
            message: "Phiên đã lưu không còn tồn tại trên server.",
          });
          return;
        }
        dispatch({ type: "REQUEST_FAILED", message: messageFromError(error) });
      }
    },
    [],
  );

  useEffect(() => {
    const threadId = initialThreadId.current;
    if (!threadId) return;
    const controller = new AbortController();
    void restoreThread(threadId, controller.signal);
    return () => controller.abort();
  }, [restoreThread]);

  useEffect(() => {
    if (!state.threadId) {
      localStorage.removeItem(STORAGE_KEY);
      return;
    }
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          version: 1,
          threadId: state.threadId,
          items: state.items,
        }),
      );
    } catch {
      // The live conversation remains usable when browser storage is unavailable.
    }
  }, [state.threadId, state.items]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [state.items, state.progress, state.error, feedbackTarget]);

  useEffect(
    () => () => {
      activeRequest.current?.abort();
      suggestionRequest.current?.abort();
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
  const busy =
    state.phase === "sending" ||
    state.phase === "resuming" ||
    state.phase === "restoring";
  const canSend = state.phase === "idle" && !activeClarification;

  useEffect(() => {
    if (state.phase !== "idle" || activeClarification) return;
    const target = state.items.findLast(
      (item): item is AssistantItem =>
        item.type === "assistant" &&
        !item.streaming &&
        Boolean(item.text.trim()),
    );
    if (!target || target.suggestionsRequested) return;
    const conversation = suggestionConversation(state.items);
    if (conversation.length < 2 || conversation.at(-1)?.role !== "assistant")
      return;

    dispatch({ type: "REQUEST_SUGGESTIONS", id: target.id });
    suggestionRequest.current?.abort();
    const controller = new AbortController();
    suggestionRequest.current = controller;
    void streamSuggestions(
      conversation,
      (event) => {
        if (event.event === "suggestions") {
          dispatch({
            type: "SET_SUGGESTIONS",
            id: target.id,
            suggestions: event.data.questions,
          });
        }
      },
      controller.signal,
    )
      .catch(() => {
        // Suggestions are optional and must never interrupt the chat flow.
      })
      .finally(() => {
        if (suggestionRequest.current === controller)
          suggestionRequest.current = null;
      });
  }, [activeClarification, state.items, state.phase]);

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

    suggestionRequest.current?.abort();
    suggestionRequest.current = null;

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
    void restoreThread(state.threadId, controller.signal).finally(() => {
      requestInFlight.current = false;
      activeRequest.current = null;
    });
  };

  const startNewConversation = () => {
    if (busy) return;
    suggestionRequest.current?.abort();
    suggestionRequest.current = null;
    localStorage.removeItem(STORAGE_KEY);
    setMessage("");
    setFeedbackTarget(null);
    dispatch({ type: "RESET" });
  };

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  const composerPlaceholder = activeClarification
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
          <button
            className="new-chat-button"
            type="button"
            disabled={busy || (!state.threadId && state.items.length === 0)}
            onClick={startNewConversation}
          >
            <span aria-hidden="true">↻</span> Cuộc trò chuyện mới
          </button>
        </div>
      </header>

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
                            onOpen={() =>
                              setFeedbackTarget({
                                assistantId: item.id,
                                product,
                              })
                            }
                          />
                        ))}
                      </div>
                    ) : null}
                    {feedbackTarget?.assistantId === item.id ? (
                      <AiFeedbackCard
                        key={feedbackTarget.product.product_id}
                        product={feedbackTarget.product}
                      />
                    ) : null}
                    {itemIndex === lastAssistantIndex && canSend ? (
                      <div className="follow-up-block">
                        <span>Bạn có thể hỏi tiếp</span>
                        <div className="suggestion-list follow-up-suggestions">
                          {(item.suggestions?.length
                            ? item.suggestions
                            : followUpSuggestions
                          ).map((suggestion) => (
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
            rows={1}
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
    </div>
  );
}
