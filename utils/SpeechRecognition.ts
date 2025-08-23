import { useEffect, useRef, useState } from "react";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import "regenerator-runtime";

export const useSpeechToText = ({
  setText,
  setIsRecording,
}: {
  setText: (text: string) => void;
  setIsRecording: (isRecording: boolean) => void;
}) => {
  const { transcript, listening, resetTranscript } = useSpeechRecognition();
  const [volume, setVolume] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const [lastTranscript, setLastTranscript] = useState("");
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Analyze audio levels (mic volume)
  const analyze = () => {
    if (analyserRef.current && dataArrayRef.current) {
      analyserRef.current.getByteFrequencyData(dataArrayRef.current);
      const sum = dataArrayRef.current.reduce((a, b) => a + b, 0);
      const avg = sum / dataArrayRef.current.length;
      setVolume(avg);
      requestAnimationFrame(analyze);
    }
  };

  // Start mic + audio context
  const startAnalyzingAudio = async () => {
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      return; // already running
    }

    const ctx = new AudioContext();
    audioContextRef.current = ctx;

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // if user stopped before initialized, clean up
    if (
      !audioContextRef.current ||
      audioContextRef.current.state === "closed"
    ) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
    analyserRef.current = audioContextRef.current.createAnalyser();
    sourceRef.current.connect(analyserRef.current);

    analyserRef.current.fftSize = 256;
    const bufferLength = analyserRef.current.frequencyBinCount;
    dataArrayRef.current = new Uint8Array(bufferLength);

    analyze();
  };

  // Stop mic + release audio context
  const stopAnalyzingAudio = () => {
    if (audioContextRef.current) {
      if (audioContextRef.current.state !== "closed") {
        audioContextRef.current.close();
      }
      audioContextRef.current = null;
    }

    if (sourceRef.current?.mediaStream) {
      sourceRef.current.mediaStream.getTracks().forEach((t) => t.stop());
    }

    setVolume(0);
  };

  // Update transcript text
  useEffect(() => {
    setText(transcript);

    if (transcript !== lastTranscript) {
      setLastTranscript(transcript);

      if (timeoutRef.current) clearTimeout(timeoutRef.current);

      timeoutRef.current = setTimeout(() => {
        SpeechRecognition.stopListening();
        stopAnalyzingAudio();
      }, 3000);
    }
  }, [transcript]);

  // Reflect recording state
  useEffect(() => {
    setIsRecording(listening);
  }, [listening]);

  // Cleanup timeout
  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      stopAnalyzingAudio();
    };
  }, []);

  // Control mic lifecycle
  useEffect(() => {
    if (listening) {
      startAnalyzingAudio();
    } else {
      stopAnalyzingAudio();
    }
  }, [listening]);

  // Start / stop button
  const toggleRecording = () => {
    if (listening) {
      SpeechRecognition.stopListening();
    } else {
      resetTranscript();
      SpeechRecognition.startListening({ continuous: true });
    }
  };

  return {
    transcript,
    listening,
    volume,
    toggleRecording,
  };
};
