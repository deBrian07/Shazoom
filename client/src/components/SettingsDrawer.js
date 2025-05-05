import React, { useState } from 'react';

const SettingsDrawer = ({ showMenu, setShowMenu, theme, setTheme, backendUrl }) => {
  const [suggestion, setSuggestion] = useState('');
  const [status, setStatus] = useState('');

  const handleSuggestionSubmit = async () => {
    if (!suggestion.includes(' - ')) {
      setStatus('Please use "Song Title - Artist" format');
      return;
    }
    const [title, artist] = suggestion.split(' - ').map(s => s.trim());
    if (!title || !artist) {
      setStatus('Both title and artist are required');
      return;
    }
    try {
      const res = await fetch(`${backendUrl}/suggest_song`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, artist }),
      });
      const data = await res.json();
      if (res.ok) {
        setStatus('Suggestion queued!');
        setSuggestion('');
      } else {
        setStatus(data.error || 'Error submitting suggestion');
      }
    } catch (err) {
      setStatus('Network error');
    }
  };

  return (
    <>
      <div className="hamburger" onClick={() => setShowMenu(true)}>☰</div>
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
            title="Auto mode"
          >
            <img src={`${process.env.PUBLIC_URL}/buttons/dark-light_mode.png`} alt="" />
          </button>
        </div>

        <div className="suggestion-container">
          <label htmlFor="song-suggestion" className="suggestion-label">Got a song we missed?</label>
          <input
            id="song-suggestion"
            type="text"
            value={suggestion}
            onChange={e => setSuggestion(e.target.value)}
            placeholder="Song Title - Artist"
            className="suggestion-input"
          />
          <button
            className="suggestion-button"
            onClick={handleSuggestionSubmit}
          >
            Submit
          </button>
          {status && <p className="suggestion-status">{status}</p>}
        </div>
      </div>
    </>
  );
};

export default SettingsDrawer;