export type Role = "user" | "assistant" | "error";

export interface Message {
  id: string;
  role: Role;
  content: string;
  /** optional HTTP‑like status coming from backend (4xx, 5xx …) */
  code?: number;
}