import { useState, useRef } from "react";
import { v4 as uuidv4 } from "uuid";
import { ChatInput } from "@/components/custom/chatinput";
import { PreviewMessage, ThinkingMessage } from "@/components/custom/message";
import { useScrollToBottom } from "@/components/custom/use-scroll-to-bottom";
import { Overview } from "@/components/custom/overview";
import { Header } from "@/components/custom/header";
import type { message } from "../../interfaces/interfaces";

const socket = new WebSocket("ws://localhost:8090");

export function Chat() {
  const [messagesContainerRef, messagesEndRef] =
    useScrollToBottom<HTMLDivElement>();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handlerRef = useRef<((e: MessageEvent) => void) | null>(null);
  const cleanup = () => {
    if (handlerRef.current) {
      socket.removeEventListener("message", handlerRef.current);
      handlerRef.current = null;
    }
  };

  // Now accepts the full payload object
  async function handleSubmit(payload: {
    query: string;
    engine: "groq";
    reasoning: boolean;
  }) {
    const { query, engine, reasoning } = payload;

    // guard: socket open + not already loading
    if (!socket || socket.readyState !== WebSocket.OPEN || isLoading) return;
    setIsLoading(true);
    cleanup();

    // add the user's message locally
    const traceId = uuidv4();
    setMessages((prev) => [
      ...prev,
      { content: query, role: "user", id: traceId },
    ]);

    // send the full JSON payload
    socket.send(JSON.stringify({ query, engine, reasoning }));

    // clear the input box
    setQuestion("");

    try {
      const onMessage = (event: MessageEvent) => {
        // end-of-stream marker
        if (event.data.includes("[END]")) {
          cleanup();
          setIsLoading(false);
          return;
        }

        // append or start assistant message
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          const updatedContent =
            last.role === "assistant"
              ? last.content + event.data
              : event.data;

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

      handlerRef.current = onMessage;
      socket.addEventListener("message", onMessage);
    } catch (err) {
      console.error("WebSocket error:", err);
      setIsLoading(false);
    }
  }

  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background">
      <Header />

      <div
        className="flex flex-col flex-1 overflow-y-scroll gap-6 pt-4"
        ref={messagesContainerRef}
      >
        {messages.length === 0 && <Overview />}
        {messages.map((m) => (
          <PreviewMessage key={m.id} message={m} />
        ))}
        {isLoading && <ThinkingMessage />}
        <div ref={messagesEndRef} className="h-[24px] w-[24px]" />
      </div>

      <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 w-full md:max-w-3xl gap-2">
        <ChatInput
          question={question}
          setQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </div>

      <h3 className="flex justify-center items-center gap-4">
        Ενδέχεται να υπάρχουν λάθη στις απαντήσεις. Πάντα να επιβεβαιώνετε τα
        αποτελέσματα.
      </h3>
    </div>
  );
}