"use client";
import { TbMessageChatbot } from "react-icons/tb";
import { useState, useEffect, useRef } from "react";
import UserMessage from "@/components/ChatBot/UserMessage";
import BotMessage from "@/components/ChatBot/BotMessage";
import TypingIndicator from "./TypingIndicator";
import Microphone from "./Microphone";
import { handleResponse } from "../../utils/utils";
import { useChatMessageContext } from "@/app/context/context";

export const ChatBot = () => {
  const [showChat, setShowChat] = useState(true);
  const [messages, setMessages] = useChatMessageContext();
  const [newMessage, setNewMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);
  console.log("Current Messages:", messages);

  const handleSendMessage = async () => {
    if (!newMessage.trim()) return;

    const userMsg = { from: "user", text: newMessage };
    setMessages((prev: any) => [...prev, userMsg]);
    setNewMessage("");
    setIsTyping(true);

    try {
      const response = await handleResponse(newMessage);
      const botMsg = { from: "bot", text: response };

      setMessages((prev: any) => [...prev, botMsg]);
    } catch (err) {
      console.error(err);
      setMessages((prev: any) => [
        ...prev,
        { from: "bot", text: "⚠️ Error fetching data. Please try again!" },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <>
      {/* Floating Chat Icon */}
      <div className="fixed right-12 bottom-6 z-50">
        <TbMessageChatbot
          size={64}
          onClick={() => setShowChat(!showChat)}
          className="p-3 bg-gradient-to-tr from-indigo-600 via-purple-600 to-indigo-700 rounded-full text-white hover:scale-110 hover:shadow-2xl transition-all duration-300 cursor-pointer shadow-lg"
        />
      </div>

      {/* Chat Window */}
      {showChat && (
        <div
          className="fixed bottom-24 right-12 w-[500px] max-w-full h-[500px] bg-white rounded-2xl border border-black flex flex-col overflow-hidden"
          style={{ boxShadow: "none" }}
        >
          {/* Header */}
          <div className="p-4 border-b border-black bg-white">
            <h2 className="font-bold text-2xl text-black flex items-center gap-2">
              <TbMessageChatbot className="text-black" />
              AI Travel Planner
            </h2>
            <p className="text-black text-sm">
              Let's plan your next adventure!
            </p>
          </div>

          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full space-y-4 text-center animate-fadeIn">
                <span className="text-6xl animate-bounce">✈️</span>
                <p className="text-gray-300 text-lg font-semibold">
                  Start the conversation to plan your next trip!
                </p>
                <p className="text-gray-400 text-sm max-w-sm">
                  Ask about destinations, weather, attractions, or travel tips!
                </p>
              </div>
            ) : (
              <>
                {messages.map((msg: any, i: any) =>
                  msg.from === "user" ? (
                    <UserMessage key={i} newMessage={msg.text} />
                  ) : (
                    <BotMessage key={i} botMessage={msg.text} />
                  )
                )}
                {isTyping && <TypingIndicator />}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input Bar */}
          <div className="p-4 border-t border-black bg-white">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="flex items-center gap-3 w-full"
            >
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                className="min-w-0 flex-1 py-2 px-4 bg-white text-black rounded-xl border border-black focus:border-black focus:outline-none placeholder-black font-medium text-base"
                placeholder="Type your destination..."
              />
              <button
                type="submit"
                disabled={!newMessage.trim()}
                className="flex items-center justify-center w-10 h-10 border border-black rounded-full bg-white text-black hover:bg-gray-100 transition-all duration-200 disabled:opacity-50"
                aria-label="Send"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  className="w-6 h-6"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 12l14-7-7 14-2-5-5-2z"
                  />
                </svg>
              </button>
              <div className="flex items-center justify-center w-10 h-10 border border-black rounded-full bg-white text-black ml-1">
                <Microphone />
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};
