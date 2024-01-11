from aiomultiprocess import Pool
from models import Frame, CameraIP

from multiprocessing import cpu_count
import asyncio
from os import path as OsPath

import asyncio
import cv2
import numpy as np
import concurrent.futures
from turbojpeg import TurboJPEG

from models import Frame
from utils import CameraIP, camera_ips
from time import sleep

def save_each_frame_process_P2(cameraIP: CameraIP, frame: Frame) -> None:
    info: dict = frame.get_last_frame(ip=cameraIP.ip)
    if info is None:
        return

    _queue, _frame = info.get('queue'), info.get('frame')
    image_path = OsPath.join(cameraIP.folder, '{}.jpg'.format(_queue))

    # engine = TurboJPEG()
    # with open(image_path, "wb") as file:
    #     file.write(engine.encode(_frame, quality=95))
    #     file.close()

    cv2.imwrite(image_path, _frame)

async def async_capture_frames_from_camera_P1(cameraIP: CameraIP, frame: Frame) -> None:
    cap = cv2.VideoCapture(cameraIP.url)
    if not cap.isOpened():
        return

    def read_frame():
        nonlocal cap

        _ret, _frame = cap.read()
        if _ret:
            frame.set_last_frame(ip=cameraIP.ip, frame=_frame)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()

        while True:
            await loop.run_in_executor(executor, read_frame)
            await loop.run_in_executor(executor, save_each_frame_process_P2, *(cameraIP, frame, ))
            await asyncio.sleep(0)

async def main_detection_process_P3(cameraIPs: list[CameraIP], frame: Frame) -> None:
    print(a)
    await asyncio.sleep(1)

async def main():
    frame = Frame()
    cameraIPs = camera_ips()
    _cpu_count = cpu_count()
    
    async with Pool(processes=_cpu_count) as pool:
        tasks = []
        for cameraIP in cameraIPs:
            tasks.append(pool.apply(async_capture_frames_from_camera_P1, (cameraIP, frame, )))
        
        tasks.append(pool.apply(main_detection_process_P3, args=(cameraIPs, frame, )))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())