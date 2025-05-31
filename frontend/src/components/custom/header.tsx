import { ThemeToggle } from "./theme-toggle";
import { InfoButton } from "./info-button";
import { EngineDropdown } from "./engine-dropdown";

interface HeaderProps {
  engine: "groq" | "opensource";
  setEngine: (engine: "groq" | "opensource") => void;
}

export const Header = ({ engine, setEngine }: HeaderProps) => {
  return (
    <header className="flex items-center justify-between px-2 sm:px-4 py-2 bg-background text-black dark:text-white w-full">
      <div className="flex items-center space-x-2">
        <ThemeToggle />
        <EngineDropdown engine={engine} setEngine={setEngine} />
      </div>
      <div className="flex items-right space-x-2">
        <InfoButton />
      </div>
    </header>
  );
};
