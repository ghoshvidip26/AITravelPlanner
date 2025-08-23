import axios from "axios";
export const handleResponse = async (message: any) => {
  const res = await axios.post(
    "http://127.0.0.1:3001/weather",
    { message },
    { headers: { "Content-Type": "application/json" } }
  );
  return res.data.answer || "No response from server.";
};
