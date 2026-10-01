# olmo-eval dashboard

Web dashboard for olmo-eval results, deployed with
[Skiff2](https://github.com/allenai/skiff-commodore) to
https://olmo-eval.allen.ai (GCP project `ai2-skiff2-olmo-eval`). Access is
limited to `@allenai.org` accounts by IAP.

- `ui/`: Vite + React + MUI/varnish. In production nginx serves the build and
  proxies `/api/*` to the API.
- `api/`: FastAPI, run as a sidecar in the same Cloud Run service.

Services are declared in `/skiff2.json`. Pushes to `main` that touch
`dashboard/` or `skiff2.json` deploy via `.github/workflows/skiff2-deploy.yml`;
PRs get a Terraform plan from `skiff2-plan.yml`.

## Local development

```bash
cd dashboard/api && uv run uvicorn app:app --reload --port 8000
cd dashboard/ui && npm install && npm run dev   # proxies /api to :8000
```
