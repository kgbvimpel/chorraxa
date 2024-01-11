import numpy as np
import json
from dataclasses import dataclass
from ipaddress import IPv4Address


@dataclass
class CameraIP:
    ip: IPv4Address
    url: str
    folder: str


class Frame:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Frame, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self._last_active_frames = {}

    def __str__(self) -> str:
        return json.dumps(self._last_active_frames)

    def set_last_frame(self, ip: str, frame: np.ndarray | None = None) -> None:
        queue: dict = self._last_active_frames.get(ip, {"queue": 0})

        self._last_active_frames[ip] = {
            "queue": queue.get("queue", 0) + 1,
            "frame": frame
        }
        return self._last_active_frames[ip]["queue"]

    def get_last_frame(self, ip: str):
        return self._last_active_frames.get(ip, None)

    def get_last_frames(self):
        return json.dumps(self._last_active_frames)
