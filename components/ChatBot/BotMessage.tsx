import { RiRobot3Line } from "react-icons/ri";
import ReactMarkdown from "react-markdown";

interface BotMessageProps {
  botMessage: any;
}

const BotMessage = ({ botMessage }: BotMessageProps) => {
  const date = new Date();
  return (
    <div className="flex w-full my-2 items-start">
      <div className="flex justify-center items-center w-8 h-8 border bg-slate-800 rounded-full  mt-auto">
        <RiRobot3Line size={20} />
      </div>
      <div className="inline-block rounded-xl bg-blue-500 p-3 text-lg text-white max-w-[75%] break-words">
        <ReactMarkdown>{botMessage}</ReactMarkdown>
        <div className="text-right">
          <span className="text-xs text-white opacity-90">
            {date.toLocaleTimeString().slice(0, 4)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default BotMessage;
