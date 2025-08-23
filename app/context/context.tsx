"use client";
import { createContext, useContext, useState, ReactNode } from "react";

const chatMessageContext = createContext<any[]>([]);
export const useChatMessageContext = () => {
  const context = useContext(chatMessageContext);
  if (!context) {
    throw new Error(
      "useChatMessageContext must be used within a ChatMessageProvider"
    );
  }
  return context;
};

export const ChatMessageProvider = ({ children }: { children: ReactNode }) => {
  const [messages, setMessages] = useState<any[]>([]);
  return (
    <chatMessageContext.Provider value={[messages, setMessages]}>
      {children}
    </chatMessageContext.Provider>
  );
};
