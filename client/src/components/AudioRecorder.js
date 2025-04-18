import React, { useState, useRef, useEffect } from 'react';
import './AudioRecorder.css';

const AudioRecorder = ({ backendUrl }) => {
  /* -------------------------------------------------- */
  /*  UI state                                          */
  /* -------------------------------------------------- */
  const [isRecording, setIsRecording] = useState(false);
  const [headerText, setHeaderText]   = useState('Tap to Shazoom');
  const [result, setResult]           = useState(null);
  const [isLoading, setIsLoading]     = useState(false);

  const [showMenu, setShowMenu]       = useState(false);
  const [theme, setTheme]             = useState('auto');           // 'light' | 'dark' | 'auto'

  /* -------------------------------------------------- */
  /*  refs                                              */
  /* -------------------------------------------------- */
  const mediaRecorderRef   = useRef(null);
  const wsRef              = useRef(null);
  const recordingTimeout   = useRef(null);
  const rippleInterval     = useRef(null);
  const buttonWrapperRef   = useRef(null);
  const mqDark             = useRef(null);                          // matchMedia listener for auto‑theme

  /* -------------------------------------------------- */
  /*  THEME handling                                    */
  /* -------------------------------------------------- */
  useEffect(() => {
    const applyTheme = mode => {
      if (mode === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {                                  // light *or* auto‑light
        document.documentElement.removeAttribute('data-theme');
      }
    };

    /* first run ------------------------------------- */
    if (theme === 'auto') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      applyTheme(prefersDark ? 'dark' : 'light');
    } else {
      applyTheme(theme);
    }

    /* listen to OS change when in auto mode --------- */
    if (theme === 'auto' && window.matchMedia) {
      mqDark.current = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = e => applyTheme(e.matches ? 'dark' : 'light');
      mqDark.current.addEventListener('change', handler);
      return () => mqDark.current.removeEventListener('change', handler);
    }
  }, [theme]);

  /* -------------------------------------------------- */
  /*  ripple helper                                     */
  /* -------------------------------------------------- */
  const spawnRipple = () => {
    if (!buttonWrapperRef.current) return;
    const el = document.createElement('div');
    el.className = 'ripple';
    buttonWrapperRef.current.appendChild(el);
    setTimeout(() => el.remove(), 1500);
  };

  /* -------------------------------------------------- */
  /*  main streaming logic                              */
  /* -------------------------------------------------- */
  const startStreaming = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      /* build WebSocket url ------------------------- */
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

        rippleInterval.current   = setInterval(spawnRipple, 600);
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
          /* ------ display result ------------------- */
          if (data.status === 'No match found after recording') {
            setResult({ result: 'No match found' });
          } else if (data.song) {
            setResult({ song: data.song, artist: data.artist });
          } else if (data.result?.song) {
            setResult({ song: data.result.song, artist: data.result.artist });
          } else if (data.error) {
            setResult({ error: data.error });
          }

          /* ------ cleanup -------------------------- */
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

  /* -------------------------------------------------- */
  /*  click handler                                     */
  /* -------------------------------------------------- */
  const handleRecordClick = () => {
    if (!isRecording) {
      setResult(null);
      setIsLoading(false);
      setHeaderText('Listening...');
      startStreaming();
    }
  };

  /* -------------------------------------------------- */
  /*  JSX                                               */
  /* -------------------------------------------------- */
  return (
    <div className="audio-recorder">
      {/* ☰ hamburger */}
      <div className="hamburger" onClick={() => setShowMenu(true)}>☰</div>

      {/* sliding drawer */}
      <div className={`drawer ${showMenu ? 'open' : ''}`}>
        <button className="close-btn" onClick={() => setShowMenu(false)}>✕</button>
        <h3 style={{ textAlign: 'center', marginTop: '2rem' }}>Settings</h3>

        <div className="mode-options">
          <button
            className={theme === 'light' ? 'active' : ''}
            onClick={() => setTheme('light')}
            title="Light mode"
          >
            <img src={`${process.env.PUBLIC_URL}/buttons/light_mode.png`} alt="" />
          </button>
          
          <button
            className={theme === 'dark' ? 'active' : ''}
            onClick={() => setTheme('dark')}
            title="Dark mode"
          >
            <img src={`${process.env.PUBLIC_URL}/buttons/dark_mode.png`} alt="" />
          </button>

          <button
            className={theme === 'auto' ? 'active' : ''}
            onClick={() => setTheme('auto')}
          >
            Auto
          </button>
        </div>
      </div>

      {/* main UI ------------------------------------- */}
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
          {result.song ? (
            <p>Recognized Song: {result.song} by {result.artist}</p>
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