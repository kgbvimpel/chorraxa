from os import path as OsPath
from time import sleep
import multiprocessing
from threading import Thread
from multiprocessing import Lock, Manager, Queue
from concurrent.futures import ProcessPoolExecutor
from models import CameraIP
import numpy as np
import cv2
from turbojpeg import TurboJPEG

from utils import camera_ips

locker = Lock()


def saving_each_frame_with_turbojpeg(image_path: str, data: np.ndarray) -> bool:
    engine = TurboJPEG()

    with locker:
        with open(image_path, "wb") as file:
            file.write(engine.encode(data, quality=95))
            file.close()
        return True


def saving_each_frame_with_cv2imwrite(image_path: str, data: np.ndarray) -> bool:
    with locker:
        result = cv2.imwrite(filename=image_path, img=data)
        return result


def connect_to_camera(cameraIP: CameraIP, active_frames: dict) -> None:
    print('Camera... {}'.format(cameraIP.ip))
    cap = cv2.VideoCapture(cameraIP.url)
    if not cap.isOpened():
        print(f"Error: Could not open camera {cameraIP}.")
        return
    print('Camera connected succesfully... {}'.format(cameraIP.ip))

    counter = 0
    while 1:
        while locker:
            ret, frame = cap.read()

            if not ret:
                print(f"Error: Could not read frame from camera {cameraIP}.")
                break

            image_path = OsPath.join(cameraIP.folder, f'{counter}.jpg')
            print(image_path)
            saving_each_frame_with_cv2imwrite(
                image_path=image_path,
                data=frame
            )
            counter += 1

            # Store the active frame in the shared dictionary
            active_frames[cameraIP.ip] = {
                "count": counter,
                "data": frame
            }

        cap.release()


def analyze_active_frame(frames: list[tuple], video_queues: Queue) -> None:
    print(frames)
    return frames


def create_an_video(tasks: Queue) -> None:
    pass


def main():
    cameraIPs: list[CameraIP] = camera_ips()

    with Manager() as manager:
        active_frames = manager.dict()
        video_queues = manager.Queue()

        with ProcessPoolExecutor(max_workers=2) as pool_connection_to_camera:
            futures = [
                pool_connection_to_camera.submit(
                    connect_to_camera,
                    cameraIP,
                    active_frames)
                for cameraIP in cameraIPs
            ]

            with ProcessPoolExecutor(max_workers=6) as pool_analyze_frames:
                while 1:
                    info = [
                        (
                            cameraIP,
                            active_frames[cameraIP]['count'],
                            # active_frames[cameraIP]['data']
                        ) for cameraIP in active_frames.keys()
                    ]
                    if info:
                        print(info)
                        pool_analyze_frames.submit(
                            analyze_active_frame,
                            info,
                            video_queues
                        )

                    if not video_queues.empty():
                        pool_analyze_frames.submit(
                            create_an_video,
                            video_queues
                        )

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        for future in futures:
            print(future.result())


if __name__ == '__main__':
    main()
