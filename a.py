import cv2
import threading


def connect_to_camera(camera_index, frames_queue):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"Error: Could not read frame from camera {camera_index}.")
            break

        frames_queue.put((camera_index, frame))

    cap.release()


# Create a queue to store frames
frames_queue = queue.Queue()

# Create thread instances for each camera
thread1 = threading.Thread(target=connect_to_camera, args=(0, frames_queue))
thread2 = threading.Thread(target=connect_to_camera, args=(1, frames_queue))

# Start the threads
thread1.start()
thread2.start()

while True:
    # Get frames from the queue and process them as needed
    if not frames_queue.empty():
        camera_index, frame = frames_queue.get()
        # Process the frame or save it to a file, etc.
        cv2.imshow(f'Camera {camera_index}', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Wait for threads to finish
thread1.join()
thread2.join()

cv2.destroyAllWindows()
