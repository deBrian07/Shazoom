import React, { useState, useRef } from 'react';
import './AudioRecorder.css';

const AudioRecorder = ({ backendUrl }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const rippleIntervalRef = useRef(null);
  const buttonWrapperRef = useRef(null);

  // Function to spawn a ripple element that expands and fades out.
  const spawnRipple = () => {
    if (!buttonWrapperRef.current) return;
    const ripple = document.createElement('div');
    ripple.className = 'ripple';
    buttonWrapperRef.current.appendChild(ripple);
    // Remove ripple after animation completes (1.5 seconds)
    setTimeout(() => {
      ripple.remove();
    }, 1500);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      setResult(null);
      setIsLoading(false);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        // When recording stops, clear the ripple spawner
        if (rippleIntervalRef.current) {
          clearInterval(rippleIntervalRef.current);
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        // Activate loading spinner while waiting for backend result
        setIsLoading(true);
        sendAudioToBackend(audioBlob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      console.log("Recording started...");
      
      // Spawn new ripple every 600ms while recording
      rippleIntervalRef.current = setInterval(() => {
        spawnRipple();
      }, 600);
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
        setIsLoading(false);
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
        setIsLoading(false);
        setResult("There was an error identifying the song.");
      });
  };

  // Compute header text based on state.
  let titleText = "Tap to Shazoom";
  if (isLoading) {
    titleText = "Shazooming...";
  } else if (isRecording) {
    titleText = "Listening...";
  }

  return (
    <div className="audio-recorder">
      <h2 className="recorder-title">{titleText}</h2>
      {/* Button-wrapper allows ripples and spinner to expand outside the button */}
      <div ref={buttonWrapperRef} className="button-wrapper">
        { !isRecording ? (
          <button 
            onClick={startRecording} 
            className={`record-button ${isLoading ? 'loading' : ''}`} 
          />
        ) : (
          <button 
            onClick={stopRecording} 
            className={`stop-button ${isLoading ? 'loading' : ''}`}
          >
            ⏹
          </button>
        )}
      </div>
      {result && (
        <div className="result">
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

export default AudioRecorder;