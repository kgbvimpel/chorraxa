from models import Frame, CameraIP
import asyncio
import cv2
import multiprocessing
import concurrent.futures
from os import path as OsPath
from utils import camera_ips


def create_video_process_P3(ip: str, queue: int) -> None:
    print('create_video_process_P3')
    return
    while True:
        continue


# async def save_frame_procces_P2(ip: str, frame: Frame) -> None:
#     image_data: dict = frame.get_last_frame(ip=ip)

#     if image_data is None:
#         return

#     queue, data = image_data[queue], image_data['frame']

#     # save_task = asyncio.to_thread(
#     #     cv2.imwrite, f"{FOLDER}/{queue}.jpg", image_data.get('frame'))
#     # await save_task

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


async def capture_frame_main_process_P1(cameraIP: CameraIP, frame: Frame):
    print('capture_frame_main_process_P1')
    cap = cv2.VideoCapture(cameraIP.url)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

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

    # tasks = []
    # try:
    #     while True:
    #         ret, frame = cap.read()
    #         queeue = frame.set_last_frame(ip=ip, frame=frame)

    #         if not ret:
    #             continue

    #         capture_task = asyncio.create_task(
    #             save_frame_procces_P2(ip=ip, frame=frame))
    #         tasks.append(capture_task)

    # except KeyboardInterrupt:
    #     print("Interrupted by user")

    # finally:
    #     # Release the camera
    #     cap.release()

    #     await asyncio.gather(*tasks)


def create_video():
    print('create_video')
    # while True:
    #     condition_met_frames = video_conn.recv()

        # Your logic to create a video goes here
        # For example, use cv2.VideoWriter


async def main(cameraIPs: list[CameraIP]):
    frame = Frame()
    
    frames_queue = multiprocessing.Queue()

    # Create a multiprocessing pipe for communication between processes
    parent_conn, child_conn = multiprocessing.Pipe()

    # Start the camera reading tasks
    read_tasks = [capture_frame_main_process_P1(cameraIP=cameraIP, frame=frame) for cameraIP in cameraIPs]  # Camera 1

    # Start the video creation process
    video_process = multiprocessing.Process(
        target=create_video, args=())
    video_process.start()

    # Start the frame analysis task
    analyze_process = multiprocessing.Process(
        target=create_video_process_P3, args=(frames_queue, 11,))
    analyze_process.start()

    print('.....................')
    # Run camera reading tasks concurrently
    asyncio.gather(*read_tasks)

    # Stop the frame analysis task
    # analyze_task.cancel()

    # Wait for the frame analysis task to finish
    # asyncio.run(analyze_task)

    # Terminate the video creation process
    video_process.terminate()
    video_process.join()


if __name__ == "__main__":
    asyncio.run(main(cameraIPs=camera_ips()))

# if __name__ == "__main__":
#     ips = utils.camera_ips()

#     asyncio.run(main_process(ips))
