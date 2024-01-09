import aiofiles
from os import path as OSPath
import numpy as np

class MetaFrame(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls] 


class FrameData:
    def __init__(self, ip: str, counter: str, data: np.ndarray) -> None:
        self.ip = ip
        self.path = counter
        self.data = data

    def img_path(self):
        return OSPath.join('frames', self.ip, f'{self.path}.jpeg')

    def info(self):
        return self.path, self.data

class Frame(metaclass=MetaFrame):
    _active_frames = {}
    _counter = 0
    
    def __init__(self, ip: str) -> None:
        self.ip: str = ip
        self._active_frames[self.ip] = None
    
    def frame_queue(self, ip: str):
        return self._counter
    
    def set_last_frame(self, ip: str, frame: None | np.ndarray=None) -> None:
        self._counter += 1

        self._active_frames[ip] = FrameData(ip=ip, counter=self._counter, data=frame)

    def get_last_frame(self, ip: str) -> None | np.ndarray:
        return self._active_frames[ip]
    
    async def save_image(self, ip: str):
        frameData: FrameData = self.get_last_frame()
        _path, _data = frameData.info()

        async with aiofiles.open(_path, mode='wb') as f:
            await f.write(_data)