from os import path as OsPath

import asyncio
import cv2
import numpy as np
import multiprocessing
from multiprocessing.connection import PipeConnection
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

    engine = TurboJPEG()
    with open(image_path, "wb") as file:
        file.write(engine.encode(_frame, quality=95))
        file.close()


async def async_capture_frames_from_camera_P1(cameraIP: CameraIP, frame: Frame) -> None:
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        print('Fock...')
        return

    def read_frame():
        nonlocal cap

        _ret, _frame = cap.read()
        if not _ret:
            frame.set_last_frame(ip=cameraIP.ip, frame=_frame)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()

        while True:
            await loop.run_in_executor(executor, read_frame)
            await loop.run_in_executor(executor, save_each_frame_process_P2, args=(cameraIP, frame, ))
            await asyncio.sleep(0)


async def analyze_active_frame_proccess_P3(cameraIP: CameraIP, frame: Frame, parent_conn: PipeConnection):
    condition_met_frames = multiprocessing.Manager().list()

    while True:
        if not frames_queue.empty():
            frame = frames_queue.get()

            # Your condition checking logic goes here
            # For example, checking if a certain color is present
            if np.any(frame == [0, 255, 0]):
                condition_met_frames.append(frame)

        await asyncio.sleep(0)  # Allow other tasks to run

        # Your logic to create a video from condition_met_frames goes here
        if len(condition_met_frames) > 0:
            video_process.send(condition_met_frames.copy())
            condition_met_frames.clear()

# Function to create a video from frames in a separate process


def create_video_from_detected_frame_P4(task_queue):
    while True:
        sleep(0)
        # condition_met_frames = video_conn.recv()

        # Your logic to create a video goes here
        # For example, use cv2.VideoWriter


async def main(cameraIPs: tuple[CameraIP]) -> None:
    frame = Frame()

    # # Start the camera reading tasks
    # frame_reading_tasks = [
    #     asyncio.create_task(
    #         async_capture_frames_from_camera_P1(
    #             cameraIP=cameraIP, frame=frame))
    #     for cameraIP in cameraIPs
    # ]

    task_queue = multiprocessing.Queue()
    additional_task_process = multiprocessing.Process(
        target=create_video_from_detected_frame_P4, args=(task_queue,))
    additional_task_process.start()

    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for cameraIP in cameraIPs:
            pool.starmap(
                analyze_active_frame_proccess_P3,
                [(cameraIP.ip, frame, task_queue),]
            )

    # frames_queue = multiprocessing.Queue()
    # # Create a multiprocessing pipe for communication between processes
    # parent_conn: PipeConnection
    # child_conn: PipeConnection = multiprocessing.Pipe()

    # # # Start the video creation process
    # video_process = multiprocessing.Process(
    #     target=create_video_from_detected_frame_P4, args=(child_conn,))
    # video_process.start()

    # # Start the frame analysis task
    # analyze_tasks = [
    #     analyze_active_frame_proccess_P3(
    #         cameraIP=cameraIP,
    #         frame=frame,
    #         parent_conn=parent_conn
    #     ) for cameraIP in cameraIPs
    # ]

    # # Stop the frame analysis task
    # analyze_frame_task.cancel()

    # Wait for the frame analysis task to finish
    # asyncio.run(analyze_frame_task)

    # # Run camera reading tasks concurrently
    # await asyncio.gather(*frame_reading_tasks)  # => Done!

    # Terminate the video creation process
    # video_process.terminate()
    # video_process.join()


if __name__ == "__main__":
    asyncio.run(main(cameraIPs=camera_ips()))
