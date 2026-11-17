# Chrome extension

Shows a LookTwice badge on top of videos while they play.

## Install

1. Start the backend on `http://localhost:8000` (see `backend/README.md`).
2. Open `chrome://extensions` and turn on Developer mode.
3. Click "Load unpacked" and select this folder.
4. Click the LookTwice icon. The popup should say "Connected".

## How it works

- `content.js` looks for videos that are playing and visible, captures a frame every 500 ms, and sends it to the backend through the service worker (`background.js`).
- The badge uses the average of the last 10 frames where a face was found, so it does not flicker.
- Badge states: likely authentic, uncertain, likely deepfake, no face detected, can't analyze, backend offline. Each state has its own icon and text, not only a colour.
- The popup lets you turn the extension on or off, change the backend URL, and see latency stats.

## Known limitations

- Videos protected by DRM or loaded from another domain without CORS cannot be read, so the badge shows "Can't analyze this video".
- The badge is not visible in fullscreen mode.
- Only the largest face in the frame is analyzed.
- The backend has to run on the same computer.
