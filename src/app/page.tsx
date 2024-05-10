"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const [username, setUsername] = useState("");

  const router = useRouter();

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      goToTodo();
    }
  };

  const goToJournal = () => {
    router.push(`/${username}/journal`);
  };

  const goToTodo = () => {
    router.push(`/${username}/todo`);
  };

  return (
    <main
      className="flex min-h-screen flex-col items-center justify-between"
      style={{
        background: "linear-gradient(to top, #B6CECE, #0099FF)",
      }}
    >
      <div className="flex m-auto">
        <input
          className="p-2 m-auto bg-transparent placeholder-gray-200 text-white outline-none"
          type="text"
          placeholder="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button className="text-sky-800 hover:italic" onClick={goToJournal}>
          journal
        </button>
        <button className="text-sky-800 hover:italic" onClick={goToTodo}>
          todo
        </button>
      </div>
    </main>
  );
}
