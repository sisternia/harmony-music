# music-frontend

Expo Router app that serves the Cover Studio interface directly through Expo/Metro at `http://localhost:8081`. The shared UI source lives under `src/`; Expo DOM components make the browser-based interface run in the Expo app.

Start the backend in one terminal (from the repository root), then start Expo in another:

```powershell
cd .\music-frontend
npm ci
npm start
```

`npm start` launches the web app at `http://localhost:8081`. For the native Expo app and QR code, use `npm run start:native`. The backend API runs separately at `http://127.0.0.1:8100`; Expo does not need a separate Vite process.

Web routes: `/` (new transcription), `/projects`, `/collections`, `/piano`, `/p/<projectId>` (song studio), and `/auth?mode=login` or `/auth?mode=register`. Browser tab titles end in `Harmony Auralis`.

For a physical phone, set `EXPO_PUBLIC_API_URL` in `.env.local` to the computer's LAN API address, such as `http://192.168.1.20:8100`. Android emulators default to `10.0.2.2`; iOS simulators use the Expo host address.

Build the web app with `npm run build`; exported files are written to `dist/`.

Color roles, audit findings, and maintenance commands are documented in [docs/color-system.md](docs/color-system.md). Edit `src/theme/tokens.json`, run `npm run theme:generate`, then `npm run theme:check`. CSS, native views, and canvas use the same tokens; DOM entry points import the theme directly for Expo/Metro.
