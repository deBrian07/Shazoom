const { app, BrowserWindow, systemPreferences } = require('electron');
const path = require('path');

async function ensureMicPermission() {
  const granted = await systemPreferences.askForMediaAccess('microphone');
  if (!granted) {
    console.warn('Microphone permission denied');
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 400,
    height: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      // preload: path.join(__dirname, 'preload.js'),
    },
  });

  if (process.env.NODE_ENV !== 'production') {
    // 👉 React dev server must be running (npm start)
    win.loadURL('http://localhost:3000');
    // optional: open DevTools
    // win.webContents.openDevTools();
  } else {
    // 👉 load the static build
    win.loadFile(path.join(__dirname, '..', 'build', 'index.html'));
  }
}

app.whenReady().then(async () => {
  await ensureMicPermission();  // ← request mic access here
  createWindow();               // ← then open your window
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
