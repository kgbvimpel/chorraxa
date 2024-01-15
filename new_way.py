from time import sleep
import multiprocessing
from threading import Thread, Lock as threadLock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque
from queue import Queue
from models import CameraIP
import cv2
from os import path as OsPath

from utils import camera_ips

threadLocker = threadLock()


def saving_each_frame(frame):
    for _ in range(2):  # For demonstration purposes, save frames for 5 iterations
        print(f'Saving each frame: {frame}...')
        sleep(1)


def connect_to_camera(cameraID: CameraIP, active_frames: dict) -> None:
    print('Camera... {}'.format(cameraID.ip))
    cap = cv2.VideoCapture(cameraID.url)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cameraID}.")
        return

    counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Could not read frame from camera {cameraID}.")
            break

        # Save frames as images
        image_filename = OsPath.join(cameraID.folder, f'{counter}.jpg')
        print(image_filename)
        cv2.imwrite(image_filename, frame)
        counter += 1

        # Store the active frame in the shared dictionary
        active_frames[cameraID.ip] = {
            "count": counter,
            "data": frame
        }

    cap.release()


def analyze_active_frame(frames):
    print('world')
    for _ in range(2):
        print('Analyze active frame...')
        sleep(2)

    print('Fokaasdasd.')
    return frames


def create_videos(frames):
    for _ in range(2):  # For demonstration purposes, create videos for 5 iterations
        print('Create a video...')
        sleep(1)


def main():
    cameraIPs = camera_ips()

    with multiprocessing.Manager() as manager:
        active_frames = manager.dict()

        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(connect_to_camera, cameraIP, active_frames)
                for cameraIP in cameraIPs
            ]

        # while True:
        #     print('Fock........')
        #     for cameraIP, active_frame in active_frames.items():
        #         print(cameraIP, active_frame.keys())
        #         # pool.submit(analyze_active_frame, cameraIP, active_frame)

        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        for future in futures:
            print(future.result())


if __name__ == '__main__':
    main()
