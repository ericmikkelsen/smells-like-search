# smells-like-search
Uses fruit fly neurons to do search

## GitHub Pages demo

This repository is configured to deploy the POC to GitHub Pages using `.github/workflows/deploy-pages.yml`.

### One-time repository settings

1. Go to **Settings → Pages**.
2. Set **Source** to **GitHub Actions**.

### Publish flow

- Push to `main` (or run the workflow manually from the Actions tab).
- The workflow compiles `assembly/flyhash.ts` into `assembly/flyhash.wasm` and deploys:
  - `index.html`
  - `FlyRAGSearchEngine.js`
  - `rag.worker.js`
  - generated `assembly/flyhash.wasm`
