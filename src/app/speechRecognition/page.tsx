"use client";
import React, { useRef, useState } from "react";
import { TalkShape } from "../[username]/page";

const SpeechRecognition = () => {
  const recognitionRef = useRef<any>(null);

  const [value, setValue] = useState("");

  const startStt = () => {
    setValue("");
    recognitionRef.current = new window.webkitSpeechRecognition();
    recognitionRef.current.start();

    recognitionRef.current.onresult = (event: any) => {
      const last = event.results[event.results.length - 1];
      if (last.isFinal) {
        setValue((v) => v + " " + event.results[0][0].transcript);
      }
    };
  };

  const stopStt = () => {
    recognitionRef.current.stop();
  };

  return (
    <div>
      <p>{value}</p>
      <button
        onMouseDown={startStt}
        onMouseUp={stopStt}
        className="bg-blue-400 hover:bg-blue-500 active:bg-red-400 rounded-full w-20 h-20 focus:outline-none flex justify-center items-center m-4"
      >
        <TalkShape />
      </button>
    </div>
  );
};

export default SpeechRecognition;
