import axios from "axios";

export const handleResponse = async (message: string): Promise<string> => {
  try {
    const res = await axios.post(
      "http://127.0.0.1:3001/weather",
      { message },
      { headers: { "Content-Type": "application/json" } }
    );

    const data = res.data;
    if (typeof data === "string") return data;
    if (data?.answer) return data.answer;
    if (data?.message) return data.message;

    return "⚠️ Sorry, I couldn't understand the server response.";
  } catch (err: any) {
    console.error("handleResponse error:", err.message);
    return "⚠️ Server error. Please try again.";
  }
};
