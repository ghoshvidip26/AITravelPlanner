import { RiRobot3Line } from "react-icons/ri";
interface BotMessageProps {
  botMessage: any;
}
const BotMessage = ({ botMessage }: BotMessageProps) => {
  return (
    <div className="flex w-full my-2">
      <div className="flex justify-center p-1 w-8 h-8 border bg-slate-800 rounded-full mr-2">
        <RiRobot3Line size={24} />
      </div>
      <div>{botMessage}</div>
    </div>
  );
};

export default BotMessage;
