"""Root entrypoint for local OpenEnv runs and Hugging Face Spaces."""

from __future__ import annotations

import os

from meverse.server.app import app as openenv_app
from meverse.server.app import main as openenv_main


def _running_in_hf_space() -> bool:
    return any(os.getenv(name) for name in ("SPACE_ID", "SPACE_AUTHOR_NAME", "HF_SPACE_ID"))


def _app_mode() -> str:
    if _running_in_hf_space():
        return "dashboard"
    return os.getenv("TRADEX_APP_MODE", "openenv").strip().lower()


if _app_mode() == "dashboard":
    from dashboard import build_app

    app = build_app()

    def main() -> None:
        port = int(os.getenv("PORT", "7860"))
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
        )
else:
    app = openenv_app
    main = openenv_main


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
