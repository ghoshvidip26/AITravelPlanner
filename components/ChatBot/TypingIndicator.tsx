const TypingIndicator = () => {
  return (
    <div className="flex items-end my-4">
      <div className="flex items-center justify-center w-13 h-8 mx-2 bg-gray-200 rounded-xl">
        <div className="w-1 h-1 mx-1 bg-gray-700 rounded-full animate-bounce [animation-delay:0s]"></div>
        <div className="w-1 h-1 mx-1 bg-gray-700 rounded-full animate-bounce [animation-delay:0.15s]"></div>
        <div className="w-1 h-1 mx-1 bg-gray-700 rounded-full animate-bounce [animation-delay:0.3s]"></div>
      </div>
    </div>
  );
};

export default TypingIndicator;
