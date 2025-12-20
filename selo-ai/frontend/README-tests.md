# SELO DSP Frontend Tests

## Prereqs (Ubuntu server)
- Node 18+ and npm 9+

## Install
```bash
cd selo-ai/frontend
npm ci
```

## Unit/UI tests (Jest/RTL)
```bash
npm test -- --watchAll=false
```

## E2E tests (Playwright)
Install browsers once:
```bash
npx playwright install --with-deps
```
Run E2E (uses playwright.config.ts):
```bash
npm run test:e2e
```
Notes:
- The config will auto-start the dev server (react-scripts) unless you set `E2E_SKIP_WEBSERVER=1`.
- To point to a running instance instead of starting a dev server, set:
  ```bash
  E2E_SKIP_WEBSERVER=1 E2E_BASE_URL="http://<frontend-host>:3000" npm run test:e2e
  ```
