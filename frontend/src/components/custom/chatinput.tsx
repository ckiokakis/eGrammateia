import { Textarea } from "../ui/textarea";
import { cx } from "classix";
import { Button } from "../ui/button";
import { ArrowUpIcon } from "./icons";
import { toast } from "sonner";
import { motion } from "framer-motion";
import { useState } from "react";

interface ChatInputProps {
    question: string;
    setQuestion: (question: string) => void;
    onSubmit: (payload: {
        api: string;
        query: string;
        engine: "cortex";
        reasoning: boolean;
    }) => void;
    isLoading: boolean;
}

const suggestedActions = [
    {
        title: "Ποιός είναι ο Πρόεδρος",
        label: "του τμήματος;",
        action: "Ποιός είναι ο Πρόεδρος του τμήματος;",
    },
    {
        title: "Ποιοί είναι οι τέσσερεις",
        label: "τομείς του τμήματος;",
        action: "Ποιοί είναι οι τέσσερεις τομείς του τμήματος;",
    },
];

export const ChatInput = ({
    question,
    setQuestion,
    onSubmit,
    isLoading,
}: ChatInputProps) => {
    const [showSuggestions, setShowSuggestions] = useState(true);
    const [reasoning, setReasoning] = useState(false);

    const handleSubmit = (text: string) => {
        onSubmit({ api: "41b9b1b5-9230-4a71-90b8-834996ff29c3", query: text, engine: "cortex", reasoning });
        setShowSuggestions(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (isLoading) {
                toast.error("Please wait for the model to finish its response!");
            } else if (question.trim().length > 0) {
                handleSubmit(question.trim());
            }
        }
    };

    return (
        <div className="relative w-full flex flex-col gap-4">
            {showSuggestions && (
                <div className="hidden md:grid sm:grid-cols-2 gap-2 w-full">
                    {suggestedActions.map((sa, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 20 }}
                            transition={{ delay: 0.05 * idx }}
                            className={idx > 1 ? "hidden sm:block" : "block"}
                        >
                            <Button
                                variant="ghost"
                                onClick={() => handleSubmit(sa.action)}
                                className="text-left border rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start"
                            >
                                <span className="font-medium">{sa.title}</span>
                                <span className="text-muted-foreground">{sa.label}</span>
                            </Button>
                        </motion.div>
                    ))}
                </div>
            )}

            <input
                type="file"
                className="fixed -top-4 -left-4 size-0.5 opacity-0 pointer-events-none"
                multiple
                tabIndex={-1}
            />

            <Textarea
                placeholder="Στείλε ένα μήνυμα..."
                className={cx(
                    "min-h-[24px] max-h-[calc(75dvh)] overflow-hidden resize-none rounded-xl text-base bg-muted"
                )}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={3}
                autoFocus
            />

            {/* <Button
                aria-disabled={!reasoning}
                className={cx(
                    "rounded-full p-1.5 h-fit absolute bottom-1.5 right-10 m-0.5 border dark:border-zinc-600 transition-colors",
                    reasoning
                        ? "bg-primary text-primary-foreground hover:bg-primary/90"
                        : "bg-gray-900 border-gray-200 opacity-50 hover:bg-gray-700 pointer-events-auto"
                )}
                onClick={() => setReasoning((r) => !r)}
            >
                Reasoning
            </Button> */}


            <Button
                className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border dark:border-zinc-600"
                onClick={() =>
                    question.trim().length > 0 && handleSubmit(question.trim())
                }
                disabled={question.trim().length === 0}
            >
                <ArrowUpIcon size={14} />
            </Button>
        </div>
    );
};