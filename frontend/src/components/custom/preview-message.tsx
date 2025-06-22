import type { Message } from "@/interfaces/message";

interface Props {
  message: Message;
}

export function PreviewMessage({ message }: Props) {
  const base = "px-4 py-2 rounded-xl max-w-[90%] whitespace-pre-line";
  if (message.role === "user")
    return (
      <div className="flex justify-end">
        <div className={`${base} bg-primary text-primary-foreground ml-auto`}>{
          message.content
        }</div>
      </div>
    );

  if (message.role === "assistant")
    return (
      <div className="flex justify-start">
        <div className={`${base} bg-muted text-muted-foreground`}>{
          message.content
        }</div>
      </div>
    );

  // error bubble (red)
  return (
    <div className="flex justify-start">
      <div
        className={
          base + " bg-red-600/20 text-red-900 border border-red-400"
        }
      >
        {message.content}
      </div>
    </div>
  );
}