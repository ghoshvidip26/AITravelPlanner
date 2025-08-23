"use client";
import { useEffect } from "react";
import { useSpeechToText } from "../../utils/SpeechRecognition";

export default function Microphone() {
  const { transcript, listening, volume, toggleRecording } = useSpeechToText({
    setText: () => {},
    setIsRecording: () => {},
  });

  useEffect(() => {
    if (transcript && !listening) {
      (async () => {
        try {
          const res = await fetch("http://127.0.0.1:3001/weather", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ newMessage: transcript }),
          });
          const data = await res.json();
          console.log("API Response:", data);
        } catch (error) {
          console.error("Error sending transcript:", error);
        }
      })();
    }
  }, [transcript, listening]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-indigo-900 via-gray-900 to-indigo-700 px-4">
      <div className="w-full max-w-md mx-auto flex flex-col items-center gap-8">
        {/* Title */}
        <div className="flex flex-col items-center mb-2">
          <span className="text-4xl font-bold text-indigo-300 mb-2">
            Voice Assistant
          </span>
          <span className="text-lg text-indigo-200">
            Speak your query and let AI help you!
          </span>
        </div>

        {/* Transcript Card */}
        <div
          className="w-full max-w-2xl min-h-[160px] p-8 rounded-3xl 
                bg-gradient-to-br from-indigo-600/20 to-indigo-400/10 
                backdrop-blur-xl shadow-2xl border border-indigo-400/30 
                flex items-center justify-center"
        >
          {transcript ? (
            <p className="text-2xl font-semibold leading-relaxed text-indigo-100 text-center whitespace-pre-wrap">
              {transcript}
            </p>
          ) : (
            <p className="text-indigo-300 italic text-center text-xl">
              <span className="text-4xl mr-3">🎙️</span>Say something...
            </p>
          )}
        </div>

        {/* Mic Button */}
        <button
          onClick={toggleRecording}
          className={`relative w-28 h-28 rounded-full flex items-center justify-center transition-all duration-300 shadow-xl border-4 border-indigo-500/40
            ${
              listening
                ? "bg-red-500 scale-110 shadow-red-500/50"
                : "bg-indigo-600 hover:scale-105 shadow-indigo-500/40"
            }
          `}
        >
          {/* Glow Ring when recording */}
          {listening && (
            <span
              className="absolute rounded-full bg-red-400 opacity-40 animate-ping"
              style={{
                width: `${120 + volume / 2}px`,
                height: `${120 + volume / 2}px`,
              }}
            ></span>
          )}
          <span className="text-5xl">🎤</span>
        </button>

        {/* Status */}
        <div className="text-base font-semibold tracking-wide mt-2">
          {listening ? (
            <span className="text-red-400 animate-pulse">Listening...</span>
          ) : (
            <span className="text-indigo-300">
              Tap the mic to start recording
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
