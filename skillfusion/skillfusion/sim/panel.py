"""Side panel served with viser on its own port (default 8081).

The Isaac Lab viser visualizer shows the 3D scene but has no camera view, log or input widgets, and
viser is the only window into this container. So the panel adds: wrist-camera frame, agent log, a
text command box (typed or produced by the voice page) and status. Import-safe without viser.
"""
from __future__ import annotations

import queue
import time

import numpy as np


class Panel:
    def __init__(self, port: int = 8081, host: str = "0.0.0.0"):
        import viser
        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        g = self.server.gui
        self.commands: "queue.Queue[str]" = queue.Queue()
        self._status = g.add_markdown("**SkillFusion** - idle")
        self._cam = g.add_image(np.zeros((240, 320, 3), np.uint8), label="Franka wrist camera")
        self._text = g.add_text("Command", "")
        self._send = g.add_button("Send")
        self._log = g.add_markdown("")
        self._lines: list[str] = []

        @self._send.on_click
        def _(_evt):
            if self._text.value.strip():
                self.commands.put(self._text.value.strip())
                self.log(f"**you:** {self._text.value.strip()}")
                self._text.value = ""

    def status(self, text: str) -> None:
        self._status.content = f"**SkillFusion** - {text}"

    def log(self, line: str) -> None:
        self._lines = (self._lines + [line])[-14:]
        self._log.content = "\n\n".join(self._lines)

    def camera(self, rgb: np.ndarray) -> None:
        self._cam.image = np.asarray(rgb, dtype=np.uint8)

    def next_command(self, timeout: float = 0.05) -> str | None:
        try:
            return self.commands.get(timeout=timeout)
        except queue.Empty:
            return None
