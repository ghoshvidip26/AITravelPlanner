import { CiUser } from "react-icons/ci";

interface UserMessageProps {
  newMessage: any;
}

const UserMessage = ({ newMessage }: UserMessageProps) => {
  const date = new Date();
  return (
    <div className="flex w-full my-2 justify-end">
      <div className="w-auto h-auto rounded-xl bg-purple-500 p-3 text-lg">
        {newMessage}
        <div className="text-right">
          <span className="text-xs text-white opacity-90">
            {date.toLocaleTimeString().slice(0, 4)}
          </span>
        </div>
      </div>

      <div className="flex justify-center p-1 w-8 h-8 border bg-slate-800 rounded-full mt-auto">
        <CiUser size={20} />
      </div>
    </div>
  );
};

export default UserMessage;
