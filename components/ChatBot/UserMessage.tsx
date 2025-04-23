import { CiUser } from "react-icons/ci";

interface UserMessageProps {
  newMessage: any;
}

const UserMessage = ({ newMessage }: UserMessageProps) => {
  return (
    <div className="flex w-full my-2">
      <div className="flex justify-center p-1 w-8 h-8 border bg-slate-800 rounded-full mr-2">
        <CiUser size={24} />
      </div>
      <div>{newMessage}</div>
    </div>
  );
};

export default UserMessage;
