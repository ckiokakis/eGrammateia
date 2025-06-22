import { useState, useRef, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { ChatInput } from "@/components/custom/chatinput";
import { PreviewMessage, ThinkingMessage } from "@/components/custom/message";
import { useScrollToBottom } from "@/components/custom/use-scroll-to-bottom";
import { Overview } from "@/components/custom/overview";
import { Header } from "@/components/custom/header";
import type { message } from "../../interfaces/interfaces";

const WS_URL = "ws://localhost:8090";

function useGuaranteedSocket() {
  const socketRef = useRef<WebSocket | null>(null);

  const getSocket = useCallback((): WebSocket => {
    const s = socketRef.current;

    // Already OPEN (1) or CONNECTING (0) → just reuse it
    if (
      s &&
      (s.readyState === WebSocket.OPEN ||
        s.readyState === WebSocket.CONNECTING)
    ) {
      return s;
    }

    // Otherwise start a brand-new connection
    s?.close();
    socketRef.current = new WebSocket(WS_URL);
    return socketRef.current;
  }, []);

  return getSocket;
}

export function Chat() {
  const getSocket = useGuaranteedSocket();

  /* auto-scroll hook */
  const [messagesContainerRef, messagesEndRef] =
    useScrollToBottom<HTMLDivElement>();

  /* component state */
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [engine, setEngine] = useState<"groq" | "opensource">("opensource");

  /* remember current streaming listener so it can be removed */
  const handlerRef = useRef<((e: MessageEvent) => void) | null>(null);
  const cleanup = () => {
    const socket = getSocket();
    if (handlerRef.current) {
      socket.removeEventListener("message", handlerRef.current);
      handlerRef.current = null;
    }
  };

  async function handleSubmit(payload: {
    api: string;
    query: string;
    engine?: "groq" | "opensource";
    reasoning: boolean;
  }) {
    if (isLoading) return; // avoid double-clicks

    const { api, query, reasoning } = payload;
    const selectedEngine = payload.engine || engine;

    setIsLoading(true);

    /* show user message immediately */
    const traceId = uuidv4();
    setMessages((prev) => [
      ...prev,
      { content: query, role: "user", id: traceId },
    ]);
    setQuestion("");

    const socket = getSocket();
    const payloadStr = JSON.stringify({
      api,
      query,
      engine: selectedEngine,
      reasoning,
    });

    const sendPayload = () => socket.send(payloadStr);

    if (socket.readyState === WebSocket.OPEN) {
      sendPayload();
    } else {
      /* CONNECTING … */
      socket.addEventListener("open", sendPayload, { once: true });
      socket.addEventListener(
        "error",
        () => {
          cleanup();
          setIsLoading(false);
          setMessages((prev) => [
            ...prev,
            {
              content:
                "Unable to connect to the server. Please check that it is running.",
              role: "error",
              id: uuidv4(),
            },
          ]);
        },
        { once: true }
      );
    }

    const onMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data?.type === "error") {
          cleanup();
          setIsLoading(false);

          setMessages((prev) => [
            ...prev,
            {
              content: `There was an error with code ${data.code}:\n${data.message}`,
              role: "error",
              id: uuidv4(),
            },
          ]);
          return;
        }
      } catch {
        /* not JSON → continue streaming */
      }

      if (event.data.includes("[END]")) {
        cleanup();
        setIsLoading(false);
        return;
      }

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        const updatedContent =
          last.role === "assistant" ? last.content + event.data : event.data;

        const newMsg = {
          content: updatedContent,
          role: "assistant" as const,
          id: traceId,
        };

        return last.role === "assistant"
          ? [...prev.slice(0, -1), newMsg]
          : [...prev, newMsg];
      });
    };

    cleanup(); // remove any stale listener first
    handlerRef.current = onMessage;
    socket.addEventListener("message", onMessage);
  }

  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      {/* top bar */}
      <Header engine={engine} setEngine={setEngine} />

      {/* messages */}
      <div
        className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4"
        ref={messagesContainerRef}
      >
        {messages.length === 0 && <Overview />}
        {messages.map((m, i) =>
          m.role === "error" ? (
            <PreviewMessage key={i} message={m} type="error" />
          ) : (
            <PreviewMessage key={i} message={m} type="normal" />
          )
        )}
        {isLoading && <ThinkingMessage />}
        <div
          ref={messagesEndRef}
          className="shrink-0 min-w-[24px] min-h-[24px]"
        />
      </div>

      {/* input */}
      <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
        <ChatInput
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
          engine={engine}
        />
      </div>

      {/* disclaimer */}
      <h3 className="flex justify-center items-center gap-4 text-sm text-muted-foreground pb-2">
        Ενδέχεται να υπάρχουν λάθη στις απαντήσεις. Πάντα να επιβεβαιώνετε τα
        αποτελέσματα.
      </h3>
    </div>
  );
}
