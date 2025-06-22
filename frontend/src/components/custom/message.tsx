import { motion } from "framer-motion";
import { cx } from "classix";
import { SparklesIcon } from "./icons";
import { Markdown } from "./markdown";
import type { message } from "../../interfaces/interfaces";
import { MessageActions } from "@/components/custom/actions";

export const PreviewMessage = ({
  message,
  type = "normal",               // ⬅ NEW: default “normal”
}: {
  message: message;
  type?: "normal" | "error";       // ⬅ NEW: accept “error”
}) => (
  <motion.div
    className="w-full mx-auto max-w-3xl px-4 group/message"
    initial={{ y: 5, opacity: 0 }}
    animate={{ y: 0, opacity: 1 }}
    data-role={message.role}
  >
    <div
      className={cx(
        "flex gap-4 rounded-xl",
        message.role === "user"
          ? "bg-zinc-700 dark:bg-muted text-white px-3 py-2 w-fit ml-auto max-w-2xl"
          : ""
      )}
    >
      {message.role === "assistant" && (
        <div className="size-8 flex items-center rounded-full justify-center ring-1 ring-border shrink-0">
          <SparklesIcon size={14} />
        </div>
      )}

      {/* ⬇ just add the red text class when type === "error" */}
      <div
        className={cx(
          "flex flex-col w-full",
          type === "error" && "text-red-500"
        )}
      >
        {message.content && (
          <div className="flex flex-col gap-4 text-left">
            <Markdown>{message.content}</Markdown>
          </div>
        )}
        {message.role === "assistant" && <MessageActions message={message} />}
      </div>
    </div>
  </motion.div>
);

export const ThinkingMessage = () => (
  <motion.div
    className="w-full mx-auto max-w-3xl px-4 group/message"
    initial={{ y: 5, opacity: 0 }}
    animate={{ y: 0, opacity: 1, transition: { delay: 0.2 } }}
    data-role="assistant"
  >
    <div className="flex gap-4 rounded-xl bg-muted px-3 py-2 w-fit ml-auto max-w-2xl">
      <div className="size-8 flex items-center rounded-full justify-center ring-1 ring-border shrink-0">
        <SparklesIcon size={14} />
      </div>
    </div>
  </motion.div>
);