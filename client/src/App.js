import React from 'react';
import AudioRecorder from './components/AudioRecorder';
import './App.css';

function App() {
  // Backend base URL (configure with REACT_APP_BACKEND_URL; falls back to the current backend IP)
  const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://47.229.32.217:5000";

  return (
    <div className="App">
      <header className="App-header">
        <h1>Shazoom</h1>
      </header>
      <main>
        <AudioRecorder backendUrl={backendUrl} />
      </main>
    </div>
  );
}

export default App;
