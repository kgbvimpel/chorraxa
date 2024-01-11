from os import path as OsPath

import asyncio
import cv2
import numpy as np
import multiprocessing
import concurrent.futures
from turbojpeg import TurboJPEG

from models import Frame
from utils import CameraIP, camera_ips
from time import sleep

from aiomultiprocess import Pool


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


# async def analyze_active_frame_proccess_P3(cameraIPs: list[CameraIP], frame: Frame):
#     print(cameraIPs)
#     await asyncio.sleep(1)
#     print(frame)

async def analyze_active_frame_proccess_P3(data):
    print(data)
    await asyncio.sleep(1)
    print(data)

    # Function to create a video from frames in a separate process


def create_video_from_detected_frame_P4(task_queue):
    while True:
        sleep(0)
        # condition_met_frames = video_conn.recv()

        # Your logic to create a video goes here
        # For example, use cv2.VideoWriter


async def main(cameraIPs: tuple[CameraIP]) -> None:
    frame = Frame()

    # Start the camera reading tasks
    frame_reading_tasks = [
        asyncio.create_task(
            async_capture_frames_from_camera_P1(
                cameraIP=cameraIP, frame=frame))
        for cameraIP in cameraIPs
    ]

    # CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        
        # Prepare data for CPU-bound tasks
        tasks_data = [i for i in range(10)]

        # Schedule CPU-bound tasks and obtain futures
        futures = [loop.run_in_executor(executor, analyze_active_frame_proccess_P3, data) for data in tasks_data]

        # Await the completion of futures
        cpu_results = await asyncio.gather(*futures)
    
    # Run camera reading tasks concurrently
    await asyncio.gather(*frame_reading_tasks)  # => Done!




if __name__ == "__main__":
    asyncio.run(main(cameraIPs=camera_ips()))

    # print(list(zip([1, 2, 3,], [4, 5, 6])))
