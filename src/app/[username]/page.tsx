"use client";
import axios from "axios";
import React, { useState } from "react";

export default function Page({ params }: { params: { username: string } }) {
  const [enabled, setEnabled] = useState(false);
  const [audioRef, setAudioRef] = useState<HTMLAudioElement | null>(null);

  const say = (text: string) => {
    axios
      .post("/api/tts", { text, speed: 0.9 }, { responseType: "blob" })
      .then((res) => {
        const blob: Blob = res.data;
        const url = URL.createObjectURL(blob);
        if (!audioRef) {
          const _audioRef = new Audio(url);
          setAudioRef(_audioRef);
          _audioRef.play();
        } else {
          audioRef.setAttribute("src", url);
          audioRef.play();
        }
      });
  };

  const handleMicrophoneClick = () => {
    setEnabled(true);
    say(`Welcome back, ${params.username}`);
  };

  return (
    <div
      className="flex min-h-screen flex-col items-center justify-between"
      style={{
        background: "linear-gradient(to top, #B6CECE, #0099FF)",
      }}
    >
      <p className="text-white font-light text-xl animate-fadeInUp text-right w-full p-3">
        <span className="text-lime-200">{params.username}</span>
      </p>
      <div className="text-white">
        <div>
          <button
            onClick={handleMicrophoneClick}
            className={`mt-10 m-auto flex items-center justify-center ${
              enabled ? "" : "bg-blue-400 hover:bg-blue-500"
            } rounded-full w-20 h-20 focus:outline-none`}
          >
            <TalkShape />
          </button>
        </div>
      </div>
      <footer></footer>
    </div>
  );
}

const TalkShape = () => {
  return (
    <svg
      viewBox="0 0 256 256"
      xmlns="http://www.w3.org/2000/svg"
      className="w-12 h-12 text-white"
    >
      <path
        fill="currentColor"
        d="M128 176a48.05 48.05 0 0 0 48-48V64a48 48 0 0 0-96 0v64a48.05 48.05 0 0 0 48 48ZM96 64a32 32 0 0 1 64 0v64a32 32 0 0 1-64 0Zm40 143.6V232a8 8 0 0 1-16 0v-24.4A80.11 80.11 0 0 1 48 128a8 8 0 0 1 16 0a64 64 0 0 0 128 0a8 8 0 0 1 16 0a80.11 80.11 0 0 1-72 79.6Z"
      />
    </svg>
  );
};
