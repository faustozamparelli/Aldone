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

export interface TodoItem {
  text: string;
  completed: boolean;
  subtasks?: TodoItem[];
}

export default function Page({ params }: { params: { username: string } }) {
  const username = params.username.replace(/%20/g, " ");

  const [enabled, setEnabled] = useState(false);
  const [audioRef, setAudioRef] = useState<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<any>(null);

  const [todos, setTodos] = useState<TodoItem[]>([
    { text: "AI Lab Project", completed: true },
    { text: "Collect Raspberri Pi at delivery center", completed: false },
    {
      text: "Pack for the trip to Rome",
      completed: false,
      subtasks: [
        { text: "Passport", completed: true },
        { text: "Clothes", completed: false },
        { text: "Toothbrush", completed: false },
        { text: "Shoes", completed: false },
      ],
    },
  ]);
  const [groceries, setGroceries] = useState<TodoItem[]>([
    { text: "Tiramisu", completed: false },
    { text: "Bananas", completed: true },
    {
      text: "Apple Pie",
      completed: false,
      subtasks: [
        { text: "Apples", completed: true },
        { text: "Pie Crust", completed: false },
        { text: "Cinnamon", completed: false },
        { text: "Butter", completed: false },
      ],
    },
    { text: "200g Avocado", completed: false },
  ]);

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
        const reqBody: ProcessVoiceRequest = { input: last[0].transcript };
        axios.post("/api/processVoice", reqBody).then((res) => {
          const { action, agentReply }: ProcessVoiceResponse = res.data;

          say(agentReply);

          if (action.addingTodo) {
            setTodos([...todos, { text: action.addingTodo, completed: false }]);
          } else if (action.retriveTodos) {
            // list todos
          }
        });
        // TODO: stop talking if user asking so
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

  const handleFinishedSpeaking = () => {};

  const say = (text: string) => {
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
    setTimeout(() => {
      say(`Welcome back, ${username}.`);
    }, 2000);
  };

  const [gradient, setGradient] = useState(["#B6CECE", "#0099FF"]);

  return (
    <>
      <div
        className={`flex min-h-screen flex-col items-center justify-between`}
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
              className={`animate-fadeInUp  mt-10 m-auto flex items-center justify-center ${
                enabled ? "" : "bg-blue-400 hover:bg-blue-500"
              } rounded-full w-20 h-20 focus:outline-none`}
            >
              {!enabled && (
                <div onClick={handleMicrophoneClick}>
                  <TalkShape />
                </div>
              )}
              {enabled && (
                <>
                  <div className="border-orange-400 p-4 rounded-xl border-4 min-h-[30vh] min-w-[30vw] text-left bg-amber-600">
                    <h3 className="font-bold text-xl">To-do:</h3>
                    {todos.map((todo, i) => (
                      <div key={i}>
                        <p>
                          [{todo.completed ? "x" : " "}] {todo.text}
                        </p>
                        {todo.subtasks &&
                          todo.subtasks.map((subtask, j) => (
                            <pre key={j}>
                              {"    " +
                                "[" +
                                (subtask.completed ? "x" : " ") +
                                "] " +
                                subtask.text}
                            </pre>
                          ))}
                      </div>
                    ))}
                  </div>
                  <div className="border-green-400 ml-4 p-4 rounded-xl border-4 min-h-[30vh] min-w-[30vw] text-left bg-teal-600">
                    <h3 className="font-bold text-xl">Groceries:</h3>
                    {groceries.map((item, i) => (
                      <div key={i}>
                        <p>
                          [{item.completed ? "x" : " "}] {item.text}
                        </p>
                        {item.subtasks &&
                          item.subtasks.map((subitem, j) => (
                            <pre key={j}>
                              {"    " +
                                "[" +
                                (subitem.completed ? "x" : " ") +
                                "] " +
                                subitem.text}
                            </pre>
                          ))}
                      </div>
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
