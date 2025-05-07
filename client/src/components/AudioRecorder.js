import React, { useState, useRef, useEffect } from 'react';
import './AudioRecorder.css';
import SettingsDrawer from './SettingsDrawer';
import RecorderButton from './RecorderButton';
import ResultDisplay from './ResultDisplay';

const AudioRecorder = ({ backendUrl }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [headerText, setHeaderText]   = useState('Tap to Shazoom');
  const [result, setResult]           = useState(null);
  const [isLoading, setIsLoading]     = useState(false);
  const [showMenu, setShowMenu]       = useState(false);
  const [theme, setTheme]             = useState('auto');

  const mediaRecorderRef   = useRef(null);
  const wsRef              = useRef(null);
  const recordingTimeout   = useRef(null);
  const rippleInterval     = useRef(null);
  const buttonWrapperRef   = useRef(null);
  const mqDark             = useRef(null);

  useEffect(() => {
    const applyTheme = mode => {
      if (mode === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
    };
    if (theme === 'auto') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      applyTheme(prefersDark ? 'dark' : 'light');
    } else {
      applyTheme(theme);
    }
    if (theme === 'auto' && window.matchMedia) {
      mqDark.current = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = e => applyTheme(e.matches ? 'dark' : 'light');
      mqDark.current.addEventListener('change', handler);
      return () => mqDark.current.removeEventListener('change', handler);
    }
  }, [theme]);

  const spawnRipple = () => {
    if (!buttonWrapperRef.current) return;
    const el = document.createElement('div');
    el.className = 'ripple';
    buttonWrapperRef.current.appendChild(el);
    setTimeout(() => el.remove(), 1500);
  };

  const startStreaming = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const wsUrl = backendUrl.replace(/^http(s?):/, m => (m === 'https:' ? 'wss:' : 'ws:')) + '/stream';
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setHeaderText('Listening...');
        setIsRecording(true);
        mediaRecorderRef.current = new MediaRecorder(stream);
        mediaRecorderRef.current.ondataavailable = e => {
          if (e.data.size && ws.readyState === WebSocket.OPEN) ws.send(e.data);
        };
        mediaRecorderRef.current.start(250);
        rippleInterval.current = setInterval(spawnRipple, 600);
        recordingTimeout.current = setTimeout(() => {
          if (mediaRecorderRef.current?.state !== 'inactive') {
            mediaRecorderRef.current.stop();
            clearInterval(rippleInterval.current);
            buttonWrapperRef.current?.querySelectorAll('.ripple').forEach(r => r.remove());
            setHeaderText('Shazooming...');
            setIsLoading(true);
          }
        }, 9000);
      };

      ws.onmessage = evt => {
        const data = JSON.parse(evt.data);
        if (
          data.song ||
          data.result ||
          data.error ||
          data.status === 'No match found after recording'
        ) {
          if (data.status === 'No match found after recording') {
            setResult({ result: 'No match found' });
          } else if (data.song) {
            setResult({ song: data.song, artist: data.artist, coverUrl: data.coverUrl, spotifyUrl: data.spotifyUrl });
          } else if (data.result?.song) {
            setResult({ song: data.result.song, artist: data.result.artist, coverUrl: data.result.coverUrl, spotifyUrl: data.result.spotifyUrl });
          } else if (data.error) {
            setResult({ error: data.error });
          }
          setHeaderText('Tap to Shazoom');
          mediaRecorderRef.current?.state !== 'inactive' && mediaRecorderRef.current.stop();
          clearTimeout(recordingTimeout.current);
          clearInterval(rippleInterval.current);
          buttonWrapperRef.current?.querySelectorAll('.ripple').forEach(r => r.remove());
          setIsRecording(false);
          setIsLoading(false);
          ws.close();
        }
      };

      ws.onerror = e => console.error('WebSocket error:', e);
      ws.onclose = () => {
        mediaRecorderRef.current?.state !== 'inactive' && mediaRecorderRef.current.stop();
        stream.getTracks().forEach(track => track.stop());
        clearTimeout(recordingTimeout.current);
        clearInterval(rippleInterval.current);
        buttonWrapperRef.current?.querySelectorAll('.ripple').forEach(r => r.remove());
        setIsRecording(false);
      };

      wsRef.current = ws;
    } catch (err) {
      console.error(err);
      alert('Unable to access microphone.');
    }
  };

  const handleRecordClick = () => {
    if (!isRecording) {
      setResult(null);
      setIsLoading(false);
      setHeaderText('Listening...');
      startStreaming();
    }
  };

  return (
    <div className="audio-recorder">
      <SettingsDrawer
        showMenu={showMenu}
        setShowMenu={setShowMenu}
        theme={theme}
        setTheme={setTheme}
        backendUrl={backendUrl}
      />

      <h2 className="recorder-title">{headerText}</h2>

      <RecorderButton
        handleRecordClick={handleRecordClick}
        isRecording={isRecording}
        isLoading={isLoading}
        buttonWrapperRef={buttonWrapperRef}
      />

      {result && <ResultDisplay result={result} />}
    </div>
  );
};

export default AudioRecorder;