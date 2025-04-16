import React, { useState, useRef } from 'react';
import './AudioRecorder.css';

const AudioRecorder = ({ backendUrl }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [headerText, setHeaderText] = useState("Tap to Shazoom");
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const mediaRecorderRef = useRef(null);
  const wsRef = useRef(null);
  const recordingTimeoutRef = useRef(null);
  const rippleIntervalRef = useRef(null);
  const buttonWrapperRef = useRef(null);

  const spawnRipple = () => {
    if (!buttonWrapperRef.current) return;
    const ripple = document.createElement('div');
    ripple.className = 'ripple';
    buttonWrapperRef.current.appendChild(ripple);
    setTimeout(() => {
      ripple.remove();
    }, 1500);
  };

  // start streaming audio using websocket
  const startStreaming = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Construct WebSocket URL
      let wsUrl;
      if (backendUrl.startsWith("https://")) {
        wsUrl = backendUrl.replace("https://", "wss://") + "/stream";
      } else if (backendUrl.startsWith("http://")) {
        wsUrl = backendUrl.replace("http://", "ws://") + "/stream";
      } else {
        wsUrl = backendUrl + "/stream";
      }
      
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      
      ws.onopen = () => {
        console.log("WebSocket connected");
        setHeaderText("Listening...");
        setIsRecording(true);

        mediaRecorderRef.current = new MediaRecorder(stream);
        mediaRecorderRef.current.ondataavailable = (event) => {
          if (event.data && event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(event.data);
          }
        };
        // sending audio chunks every 250ms.
        mediaRecorderRef.current.start(250);
        
        // ripple effect every 600ms.
        rippleIntervalRef.current = setInterval(() => {
          spawnRipple();
        }, 600);
        
        // stop recording after 9s
        recordingTimeoutRef.current = setTimeout(() => {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
            mediaRecorderRef.current.stop();
            // Stop ripple and set loading state.
            clearInterval(rippleIntervalRef.current);
            if (buttonWrapperRef.current) {
              const ripples = buttonWrapperRef.current.querySelectorAll('.ripple');
              ripples.forEach(r => r.remove());
            }
            setHeaderText("Shazooming...");
            setIsLoading(true);
          }
        }, 9000);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("WebSocket message:", data);
          if (data.song || data.result || data.error || (data.status && data.status === "No match found after recording")) {
            if (data.status && data.status === "No match found after recording") {
              setResult({ result: "No match found" });
              setHeaderText("Tap to Shazoom");
            } else {
              setResult(data);
              setHeaderText("Match found");
              setTimeout(() => {
                setHeaderText("Tap to Shazoom");
              }, 3000);
            }
            if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
              mediaRecorderRef.current.stop();
            }
            clearTimeout(recordingTimeoutRef.current);
            clearInterval(rippleIntervalRef.current);
            if (buttonWrapperRef.current) {
              const ripples = buttonWrapperRef.current.querySelectorAll('.ripple');
              ripples.forEach(r => r.remove());
            }
            ws.close();
            setIsRecording(false);
            setIsLoading(false);
            return;
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      };
      
      ws.onerror = (e) => {
        console.error("WebSocket error:", e);
      };
      
      ws.onclose = () => {
        console.log("WebSocket closed");
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
        clearTimeout(recordingTimeoutRef.current);
        clearInterval(rippleIntervalRef.current);
        if (buttonWrapperRef.current) {
          const ripples = buttonWrapperRef.current.querySelectorAll('.ripple');
          ripples.forEach(r => r.remove());
        }
        setIsRecording(false);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Unable to access microphone. Please grant permission.");
    }
  };

  // Clicking the record button starts streaming if not already recording.
  const handleRecordClick = () => {
    if (!isRecording) {
      setResult(null);
      setIsLoading(false);
      setHeaderText("Listening...");
      startStreaming();
    }
  };

  return (
    <div className="audio-recorder">
      <h2 className="recorder-title">{headerText}</h2>
      <div ref={buttonWrapperRef} className="button-wrapper">
        <button
          onClick={handleRecordClick}
          className={`record-button ${isRecording ? 'recording' : ''} ${isLoading ? 'loading' : ''}`}
          disabled={isRecording}
        />
      </div>
      {result && (
        <div className="result">
          {result.status === "Match found" && result.result ? (
            <p>
              Recognized Song: {result.result.song} by {result.result.artist}
            </p>
          ) : result.error ? (
            <p>Error: {result.error}</p>
          ) : (
            <p>No match found</p>
          )}
        </div>
      )}
    </div>
  );
};

export default AudioRecorder;