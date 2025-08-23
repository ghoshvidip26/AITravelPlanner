"use client";
import { useEffect } from "react";
import { useSpeechToText } from "../../utils/SpeechRecognition";
import { handleResponse } from "../../utils/utils";
import { useChatMessageContext } from "@/app/context/context";

export default function Microphone() {
  const [messages, setMessages] = useChatMessageContext();
  const { transcript, listening, volume, toggleRecording } = useSpeechToText({
    setText: () => {},
    setIsRecording: () => {},
  });

  useEffect(() => {
    if (transcript && !listening) {
      (async () => {
        try {
          setMessages((prev: any) => [
            ...prev,
            { from: "user", text: transcript },
          ]);

          const res = await handleResponse(transcript);
          setMessages((prev: any) => [...prev, { from: "bot", text: res }]);
        } catch (error) {
          console.error("Error sending transcript:", error);
          setMessages((prev: any) => [
            ...prev,
            { from: "bot", text: "⚠️ Error fetching data. Please try again!" },
          ]);
        }
      })();
    }
  }, [transcript, listening]);

  return (
    <div className="flex flex-col items-center gap-3">
      <button
        onClick={toggleRecording}
        className={`relative w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 shadow-md border-2 border-indigo-500/40
          ${
            listening
              ? "bg-red-500 scale-110 shadow-red-500/50"
              : "bg-indigo-600 hover:scale-105 shadow-indigo-500/40"
          }
        `}
      >
        {listening && (
          <span
            className="absolute rounded-full bg-red-400 opacity-40 animate-ping"
            style={{
              width: `${70 + volume / 2}px`,
              height: `${70 + volume / 2}px`,
            }}
          ></span>
        )}
        <span className="text-2xl">🎤</span>
      </button>
    </div>
  );
}
