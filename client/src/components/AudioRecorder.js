import React, { useState, useRef } from 'react';
import './AudioRecorder.css';

const AudioRecorder = ({ backendUrl }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [result, setResult] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      setResult(null);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        sendAudioToBackend(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      console.log("Recording started...");
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Unable to access microphone. Please grant permission.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      console.log("Recording stopped.");
    }
  };

  const sendAudioToBackend = (audioBlob) => {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");

    fetch(`${backendUrl}/identify`, {
      method: 'POST',
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("Server response:", data);
        if (data.song && data.artist) {
          setResult(`Recognized Song: ${data.song} by ${data.artist}`);
        } else if (data.result) {
          setResult(data.result);
        } else if (data.error) {
          setResult(`Error: ${data.error}`);
        }
      })
      .catch((err) => {
        console.error("Error sending audio data:", err);
        setResult("There was an error identifying the song.");
      });
  };

  return (
    <div className="audio-recorder">
      <h2 className={`recorder-title ${isRecording ? 'hidden' : ''}`}>Tap to Shazam</h2>
      { !isRecording ? (
        <button onClick={startRecording} className="record-button" />
      ) : (
        <button onClick={stopRecording} className="stop-button">
          ⏹
        </button>
      )}
      {result && (
        <div className="result">
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

export default AudioRecorder;