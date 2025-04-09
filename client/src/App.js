import React from 'react';
import AudioRecorder from './components/AudioRecorder';
import './App.css';

function App() {
  // Replace <YOUR_BACKEND_URL> with the actual URL of your Flask backend, e.g., "http://47.229.136.34:5000"
  const backendUrl = "http://47.229.136.34:5000";

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
