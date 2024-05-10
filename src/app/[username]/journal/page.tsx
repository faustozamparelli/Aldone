"use client";
import axios from "axios";
import React, { useEffect, useRef, useState } from "react";
import {
  ProcessVoiceRequest,
  ProcessVoiceResponse,
} from "../../api/processVoice/route";

declare global {
  interface Window {
    webkitSpeechRecognition: any;
  }
}

export default function Page({ params }: { params: { username: string } }) {
  const username = params.username.replace(/%20/g, " ");

  const [enabled, setEnabled] = useState(false);
  const [audioRef, setAudioRef] = useState<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<any>(null);

  const [listening, setListening] = useState(false);

  const [todos, setTodos] = useState<string[]>([]);

  const startRecording = () => {
    recognitionRef.current = new window.webkitSpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;

    recognitionRef.current.onresult = (event: any) => {
      const results = Array.from(
        { length: event.results.length },
        (_, i) => i
      ).map((i) => event.results[i][0].transcript);

      const last = event.results[event.results.length - 1];

      if (last.isFinal) {
        if (listening) {
          const reqBody: ProcessVoiceRequest = { input: results };
          axios.post("/api/processVoice", reqBody).then((res) => {
            const { action, agentReply }: ProcessVoiceResponse = res.data;
          });
        } else {
          // stop talking if user asking so
        }

        setTodos(results);
      }
    };

    recognitionRef.current.start();
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  };

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  const handleFinishedSpeaking = () => {
    setListening(true);
  };

  const say = (text: string) => {
    setListening(false);

    axios
      .post("/api/tts", { text, speed: 0.9 }, { responseType: "blob" })
      .then((res) => {
        const blob: Blob = res.data;
        const url = URL.createObjectURL(blob);
        if (!audioRef) {
          const _audioRef = new Audio(url);
          setAudioRef(_audioRef);

          _audioRef.addEventListener("ended", () => {
            handleFinishedSpeaking();
          });

          _audioRef.play();
        } else {
          audioRef.setAttribute("src", url);
          audioRef.play();
        }
      });
  };

  const handleMicrophoneClick = () => {
    setEnabled(true);
    startRecording();
    // setTimeout(() => {
    //   say(`Welcome back, ${username}.`);
    //   setTimeout(() => {
    //     say(`Tell me ""everything""`);
    //   }, 2000);
    // }, 2000);
  };

  const [gradient, setGradient] = useState(["#B6CECE", "#0099FF"]);

  return (
    <>
      <div
        className={`flex min-h-screen flex-col items-center justify-between ${
          enabled ? "cursor-none" : ""
        }`}
        style={{
          background: `linear-gradient(to top, ${gradient[0]}, ${gradient[1]})`,
        }}
      >
        <p className="text-white font-light text-xl animate-fadeInUp text-right w-full p-3">
          {!enabled && <span className="text-lime-200">{username}</span>}
        </p>
        <div className="text-white text-transparent">
          <div>
            <button
              onClick={handleMicrophoneClick}
              className={`animate-fadeInUp  mt-10 m-auto flex items-center justify-center ${
                enabled ? "cursor-none" : "bg-blue-400 hover:bg-blue-500"
              } rounded-full w-20 h-20 focus:outline-none`}
            >
              {!enabled && <TalkShape />}
              {enabled && (
                <>
                  <div className="border-blue-600 p-8 rounded-xl border-8 min-h-[30vh] min-w-[30vw] text-left">
                    <h3 className="font-bold text-xl">To-do:</h3>
                    {todos.map((todo, i) => (
                      <p key={i}> [ ] {todo}</p>
                    ))}
                  </div>
                </>
              )}
            </button>
          </div>
        </div>
        <footer className="min-h-[10vh]"></footer>
      </div>
    </>
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
