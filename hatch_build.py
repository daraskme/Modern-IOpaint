"""Hatch build checks for distribution-only assets."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Reject wheels whose frontend has not been built."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name != "wheel":
            return

        frontend_dist = Path(self.root) / "web_app" / "dist"
        if not frontend_dist.is_dir() or not (frontend_dist / "index.html").is_file():
            raise RuntimeError(
                "Cannot build the modern-iopaint wheel: web_app/dist is missing "
                "or incomplete. Build the frontend first with `cd web_app`, "
                "`npm ci`, and `npm run build`, then run the Hatch wheel build "
                "again. Hatch force-includes web_app/dist directly; do not copy "
                "it into modern_iopaint manually."
            )
