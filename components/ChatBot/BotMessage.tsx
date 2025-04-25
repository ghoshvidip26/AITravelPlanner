import { RiRobot3Line } from "react-icons/ri";
import ReactMarkdown from "react-markdown";
interface BotMessageProps {
  botMessage: any;
}
const BotMessage = ({ botMessage }: BotMessageProps) => {
  return (
    <div className="flex w-full my-2 flex-col">
      <div className="flex justify-center p-1 w-8 h-8 border bg-slate-800 rounded-full mr-2">
        <RiRobot3Line size={24} />
      </div>
      <ReactMarkdown>{botMessage}</ReactMarkdown>
    </div>
  );
};

export default BotMessage;
