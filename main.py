from models import Frame
import asyncio
import cv2
import utils
import multiprocessing


def create_video_process_P3(ip: str, queue: int) -> None:
    while True:
        continue


async def save_frame_procces_P2(ip: str, frame: Frame) -> None:
    image_data: dict = frame.get_last_frame(ip=ip)

    if image_data is None:
        return

    queue, data = image_data[queue], image_data['frame']

    save_task = asyncio.to_thread(
        cv2.imwrite, f"{FOLDER}/{queue}.jpg", image_data.get('frame'))
    await save_task


async def capture_frame_main_process_P1(frame: Frame, ip: str):
    cap = cv2.VideoCapture()
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tasks = []
    try:
        while True:
            ret, frame = cap.read()
            queeue = frame.set_last_frame(ip=ip, frame=frame)

            if not ret:
                continue

            capture_task = asyncio.create_task(
                save_frame_procces_P2(ip=ip, frame=frame))
            tasks.append(capture_task)

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Release the camera
        cap.release()

        await asyncio.gather(*tasks)


def create_video(video_conn):
    while True:
        condition_met_frames = video_conn.recv()

        # Your logic to create a video goes here
        # For example, use cv2.VideoWriter


def main():
    frames_queue = multiprocessing.Queue()

    # Create a multiprocessing pipe for communication between processes
    parent_conn, child_conn = multiprocessing.Pipe()

    # Start the camera reading tasks
    read_tasks = [capture_frame_main_process_P1(0, frames_queue),  # Camera 0
                  capture_frame_main_process_P1(1, frames_queue)]  # Camera 1

    # Start the video creation process
    video_process = multiprocessing.Process(
        target=create_video, args=(child_conn,))
    video_process.start()

    # Start the frame analysis task
    analyze_task = create_video_process_P3(frames_queue, parent_conn)

    # Run camera reading tasks concurrently
    asyncio.run(asyncio.gather(*read_tasks))

    # Stop the frame analysis task
    analyze_task.cancel()

    # Wait for the frame analysis task to finish
    asyncio.run(analyze_task)

    # Terminate the video creation process
    video_process.terminate()
    video_process.join()


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     ips = utils.camera_ips()

#     asyncio.run(main_process(ips))
