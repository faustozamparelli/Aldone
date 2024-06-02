"use client";
import axios from "axios";
import React, { useEffect, useRef, useState } from "react";
import {
  QuestionClassifierRequest,
  QuestionClassifierResponse,
} from "@/app/api/questionClassifier/route";
import {
  ConversationalAgentRequest,
  ConversationalAgentResponse,
} from "@/app/api/conversationalAgent/route";
import {
  TodoClassifierRequest,
  TodoClassifierResponse,
} from "@/app/api/todoClassifier/route";
import { TodoExtractorRequest } from "@/app/api/todoExtractor/route";

declare global {
  interface Window {
    webkitSpeechRecognition: any;
  }
}

const DEBUGGING_WITH_TEXT = true;

export interface TodoItem {
  text: string;
  completed: boolean;
  subtasks?: TodoItem[];
  id: string;
}

export default function Page({ params }: { params: { username: string } }) {
  const username = params.username.replace(/%20/g, " ");

  const [enabled, setEnabled] = useState(false);
  const [audioRef, setAudioRef] = useState<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<any>(null);

  const [cameraEnabled, setCameraEnabled] = useState(false);

  const [todos, setTodos] = useState<TodoItem[]>([
    { text: "AI Lab Project", completed: true, id: "ai_lab_project" },
    { text: "Big Data Project", completed: false, id: "big_data_project" },
    {
      text: "Collect Raspberri Pi at delivery point",
      completed: false,
      id: "raspi",
    },
    {
      text: "Pack for the trip to Rome",
      completed: false,
      id: "rome_trip_pack",
      subtasks: [
        { text: "Passport", completed: true, id: "passport" },
        { text: "Clothes", completed: false, id: "clothes" },
        { text: "Toothbrush", completed: false, id: "toothbrush" },
        { text: "Shoes", completed: false, id: "shoes" },
      ],
    },
  ]);
  const [groceries, setGroceries] = useState<TodoItem[]>([
    { text: "Tiramisu", completed: false, id: "tiramisu" },
    { text: "Yogurt Bowl", completed: false, id: "yogurt_bowl" },
    { text: "Bananas", completed: true, id: "bananas" },
    {
      text: "Apple Pie",
      completed: false,
      id: "apple_pie",
      subtasks: [
        { text: "Apples", completed: true, id: "apples" },
        { text: "Pie Crust", completed: false, id: "pie_crust" },
        { text: "Cinnamon", completed: false, id: "cinnamon" },
        { text: "Butter", completed: false, id: "butter" },
      ],
    },
    { text: "200g Avocado", completed: false, id: "avocado" },
  ]);

  const processVoiceInputText = (voiceInput: string) => {
    const reqBody: QuestionClassifierRequest = { input: voiceInput };

    axios.post("/api/questionClassifier", reqBody).then((res) => {
      const { isQuery, agentReply }: QuestionClassifierResponse = res.data;

      say(agentReply);

      if (isQuery) {
        const conversationalAgentRequest: ConversationalAgentRequest = {
          text: voiceInput,
          groceryList: groceries,
          todoList: todos,
        };
        axios
          .post("/api/conversationalAgent", conversationalAgentRequest)
          .then((res) => {
            const { narration, explodingTodo }: ConversationalAgentResponse =
              res.data;
            say(narration);

            if (explodingTodo) {
              const { category, id, subtasks } = explodingTodo;

              const setBucket =
                category === "grocery" ? setGroceries : setTodos;

              setBucket((prev) => {
                const index = prev.findIndex((item) => item.id === id);
                prev[index].subtasks = subtasks;
                return prev;
              });
            }
          });
      } else {
        const todoClassifierRequest: TodoClassifierRequest = {
          input: voiceInput,
        };

        axios.post("/api/todoClassifier", todoClassifierRequest).then((res) => {
          const { category }: TodoClassifierResponse = res.data;

          const listCategory =
            category === "shopping_list" || category === "shopping_list_update"
              ? "grocery"
              : "todo";

          const setBucket =
            listCategory === "grocery" ? setGroceries : setTodos;

          const todoExtractorRequest: TodoExtractorRequest = {
            text: voiceInput,
          };

          axios.post("/api/todoExtractor", todoExtractorRequest).then((res) => {
            const { text, id } = res.data;
            setBucket((prev) => [...prev, { text, completed: false, id }]);
          });
        });
      }
    });
  };

  const startRecording = () => {
    recognitionRef.current = new window.webkitSpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;

    recognitionRef.current.onresult = (event: any) => {
      const last = event.results[event.results.length - 1];
      if (last.isFinal) {
        const voiceInput = last[0].transcript;
        processVoiceInputText(voiceInput);
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

  const handleCameraClick = () => {
    setCameraEnabled(true);
  };
  const [seenFoods, setSeenFoods] = useState<string[]>([]);

  const handleStopCameraClick = () => {
    setCameraEnabled(false);
    axios.post("/api/readSeenFoods").then((res) => {
      const { foods } = res.data;
      setSeenFoods(foods);
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

  const [chatTextInput, setChatTextInput] = useState("");

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
          </button>
          {(enabled || DEBUGGING_WITH_TEXT) && (
            <div className="flex">
              <div>
                <div className="border-orange-400 p-4 rounded-xl border-4 min-h-[30vh] min-w-[30vw] text-left bg-amber-600">
                  <h3 className="font-bold text-xl">To-do:</h3>
                  {todos.map((todo, i) => (
                    <div key={i}>
                      <p>
                        [{todo.completed ? "x" : " "}]
                        <span className={todo.completed ? "line-through" : ""}>
                          {todo.text}
                        </span>
                      </p>
                      {todo.subtasks &&
                        todo.subtasks.map((subtask, j) => (
                          <p key={j} className="ml-4">
                            {"[" + (subtask.completed ? "x" : " ") + "] "}
                            <span
                              className={
                                subtask.completed ? "line-through" : ""
                              }
                            >
                              {subtask.text}
                            </span>
                          </p>
                        ))}
                    </div>
                  ))}
                </div>
              </div>
              <div className="flex flex-col items-center">
                <div className="border-green-400 ml-4 p-4 rounded-xl border-4 min-h-[30vh] min-w-[30vw] text-left bg-teal-600">
                  <h3 className="font-bold text-xl">Shopping List:</h3>
                  {groceries.map((item, i) => (
                    <div key={i}>
                      <p>
                        [{item.completed ? "x" : " "}]
                        <span className={item.completed ? "line-through" : ""}>
                          {item.text}
                        </span>
                      </p>
                      {item.subtasks &&
                        item.subtasks.map((subitem, j) => (
                          <p key={j} className="ml-4">
                            {"[" + (subitem.completed ? "x" : " ") + "] "}
                            <span
                              className={
                                subitem.completed ? "line-through" : ""
                              }
                            >
                              {subitem.text}
                            </span>
                          </p>
                        ))}
                    </div>
                  ))}
                </div>

                {cameraEnabled ? (
                  <div className="max-w-[30vw] flex-col flex items-center">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={"http://localhost:3001/webcam"}
                      alt="webcam"
                      className="rounded-lg shadow-lg mt-8 mx-8"
                    />
                    <button
                      className="w-12 h-12 rounded-full bg-lime-400 bg-opacity-50 hover:bg-lime-500 focus:outline-none flex items-center justify-center m-8 p-8 border-4 border-lime-400 shadow-lg"
                      onClick={handleStopCameraClick}
                    >
                      <p>⏹</p>
                    </button>
                  </div>
                ) : (
                  <div>
                    <button
                      className="w-12 h-12 rounded-full bg-lime-400 bg-opacity-50 hover:bg-lime-500 focus:outline-none flex items-center justify-center m-8 p-8 border-4 border-lime-400 shadow-lg"
                      onClick={handleCameraClick}
                    >
                      <p>📸</p>
                    </button>
                    <p className="text-center">
                      {seenFoods.length > 0 && JSON.stringify(seenFoods)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {DEBUGGING_WITH_TEXT && (
            <div className="p-4 m-auto flex justify-center items-center">
              <textarea
                value={chatTextInput}
                onChange={(e) => setChatTextInput(e.target.value)}
                className="border-2 border-gray-400 p-1 rounded-lg focus:outline-none m-2 w-60 min-w-96 text-black"
              />
              <div>
                <button
                  onClick={() => {
                    processVoiceInputText(chatTextInput);
                    setChatTextInput("");
                  }}
                  className="bg-blue-400 hover:bg-blue-500 p-1 rounded-lg text-white focus:outline-none m-2"
                >
                  submit
                </button>
              </div>
            </div>
          )}
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
