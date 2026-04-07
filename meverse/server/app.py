"""FastAPI application for the market surveillance environment."""

import os

from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import SurveillanceAction, SurveillanceObservation
    from .meverse_environment import MarketSurveillanceEnvironment
except (ModuleNotFoundError, ImportError, ValueError):
    from models import SurveillanceAction, SurveillanceObservation
    from server.meverse_environment import MarketSurveillanceEnvironment


app = create_app(
    MarketSurveillanceEnvironment,
    SurveillanceAction,
    SurveillanceObservation,
    env_name="amm-market-surveillance",
    max_concurrent_envs=1,
)


def _running_in_hf_space() -> bool:
    return any(os.getenv(name) for name in ("SPACE_ID", "SPACE_AUTHOR_NAME", "HF_SPACE_ID"))


# Only redirect / to /docs when NOT in a HF Space.
# In Space mode, Gradio is mounted at / by app.py.
if not _running_in_hf_space():
    @app.get("/")
    def root():
        """Redirect root to the API docs."""
        return RedirectResponse(url="/docs")


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
