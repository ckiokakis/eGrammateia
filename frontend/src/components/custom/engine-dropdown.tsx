import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";

interface EngineDropdownProps {
  engine: "groq" | "opensource";
  setEngine: (engine: "groq" | "opensource") => void;
}

export function EngineDropdown({ engine, setEngine }: EngineDropdownProps) {
  return (
    <div className="relative">
      <select
        id="engine-select"
        value={engine}
        onChange={(e) => setEngine(e.target.value as "groq" | "opensource")}
        className={cn(
          buttonVariants({ variant: "outline", size: "default" }),
          "pr-10 pl-3 appearance-none bg-background border text-gray-600 dark:text-gray-200"
        )}
      >
        <option value="groq">Groq</option>
        <option value="opensource">Opensource</option>
      </select>
      <ChevronDown className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 pointer-events-none text-muted-foreground" />
    </div>
  );
}
