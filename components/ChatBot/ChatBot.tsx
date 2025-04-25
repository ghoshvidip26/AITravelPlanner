"use client";
import { TbMessageChatbot } from "react-icons/tb";
import { useState } from "react";
import UserMessage from "@/components/ChatBot/UserMessage";
import BotMessage from "@/components/ChatBot/BotMessage";
import axios from "axios";
import TypingIndicator from "./TypingIndicator";

export const ChatBot = () => {
  const [showChat, setShowChat] = useState(true);
  const [messages, setMessages] = useState<any>([]);
  const [newMessage, setNewMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSendMessage = async () => {
    if (!newMessage.trim()) return;

    const userMsg = { from: "user", text: newMessage };
    setMessages((prev: any) => [...prev, userMsg]);
    setNewMessage("");
    setIsTyping(true);

    try {
      const res = await axios.post(
        "http://127.0.0.1:3001/weather",
        {
          newMessage: newMessage,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const response = res.data.response;
      const botMsg = { from: "bot", text: response };

      setMessages((prev: any) => [...prev, botMsg]);
    } catch (err) {
      console.error(err);
      setMessages((prev: any) => [
        ...prev,
        { from: "bot", text: "Error fetching data. Please try again!" },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <>
      <TbMessageChatbot
        size={64}
        onClick={() => setShowChat(!showChat)}
        className="fixed right-12 bottom-[calc(1rem)] hover:cursor-pointer"
      />
      {showChat && (
        <div className="fixed right-12 bottom-[calc(5rem)] hover:cursor-pointer p-5 shadow-md shadow-amber-50 h-full w-96">
          <div className="flex flex-col h-full">
            <div>
              <h2 className="font-semibold text-lg tracking-tight">Chatbot</h2>
              <p className="text-white">AI Travel planner</p>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 px-1">
              {messages.length === 0 ? (
                <p className="text-gray-400 text-center mt-10">
                  Start the conversation to plan your next trip ✈️
                </p>
              ) : (
                <>
                  {messages.map((msg: any, index: number) =>
                    msg.from === "user" ? (
                      <UserMessage key={index} newMessage={msg.text} />
                    ) : (
                      <BotMessage key={index} botMessage={msg.text} />
                    )
                  )}
                  {isTyping && <TypingIndicator />}{" "}
                </>
              )}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="flex flex-row mt-2 space-x-2"
            >
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                className="py-2.5 border-purple-700 border-2 px-4 w-full rounded-lg text-base focus:outline-none"
                placeholder="Type your destination..."
              />
              <button
                type="submit" // Ensures button works as form submit
                className="mt-2 py-2 px-4 bg-purple-600 text-white rounded-lg"
              >
                Send
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
};
