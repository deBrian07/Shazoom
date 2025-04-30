import React from 'react';

const SettingsDrawer = ({ showMenu, setShowMenu, theme, setTheme }) => (
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
    </div>
  </>
);

export default SettingsDrawer;