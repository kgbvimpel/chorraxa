from os import path as OsPath
from time import sleep
import multiprocessing
from multiprocessing import Lock
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
    while True:
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
    cameraIPs: list[CameraIP] = camera_ips()

    with multiprocessing.Manager() as manager:
        active_frames = manager.dict()

        with ProcessPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(connect_to_camera, cameraIP, active_frames)
                for cameraIP in cameraIPs
            ]
            print()

            while True:
                info = [
                    (
                        cameraIP,
                        active_frames[cameraIP]['count'],
                        # active_frames[cameraIP]['data']
                    ) for cameraIP in active_frames.keys()
                ]
                if info:
                    print(info)
                    # pool.submit(analyze_active_frame, cameraIP, active_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        for future in futures:
            print(future.result())


if __name__ == '__main__':
    main()
