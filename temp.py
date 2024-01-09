
from warnings import filterwarnings
filterwarnings('ignore') 

import os
import sys
import cv2
import json
import time
import torch
import threading
import queue
import logging
import traceback
# from line import process_streams, colors
# from detection import process_frame
from violation import ViolationDetector
from collections import OrderedDict
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor
# from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE


# saved_batches = []
# Function to read lines info from file
def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines_info = json.load(file)
    return lines_info

# Function to write lines info to file
def write_lines_to_file(file_path, lines_info):
    with open(file_path, 'w') as file:
        json.dump(lines_info, file)

# Function to get RTSP URL from IP address


# Function to get or generate lines info
def get_lines_info(rtsp_url, ip_address):
    line_file_path = f'line_{ip_address}.txt'
    if os.path.exists(line_file_path):
        return read_lines_from_file(line_file_path)
    else:
        lines_info = process_streams([rtsp_url])
        write_lines_to_file(line_file_path, lines_info[0])
        return lines_info[0]

frame_save_queue = queue.Queue()

def save_frame_worker(frame_batch, saved_batches, base_path, folder_range, max_frames):
    base_path = Path(base_path)
    for frame_id, frame in frame_batch:
        folder_num = frame_id // folder_range
        folder_path = base_path / f"{folder_num * folder_range}-{(folder_num + 1) * folder_range - 1}_{ip_address}"
        folder_path.mkdir(parents=True, exist_ok=True)
        file_path = folder_path / f"{frame_id}.jpg"
        try:
            jpeg = TurboJPEG()
            encoded_image = jpeg.encode(frame, quality=100)
            with open(file_path, 'wb') as output_file:
                output_file.write(encoded_image)
        except Exception as e:
            logging.error("checking saving violation video: %s", traceback.format_exc())  # Logs the traceback
        # cv2.imwrite(str(file_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Manage saved batches and folders
    saved_batches.add(str(folder_path))  # Use set for efficient lookup
    if len(saved_batches) * folder_range > max_frames:
        oldest_folder = Path(saved_batches.pop())  # Assumes FIFO insertion
        for file in oldest_folder.iterdir():
            file.unlink()
        oldest_folder.rmdir()

def frame_saving_thread():
    frame_batch = []
    current_batch_start = 0
    batch_size = 15  
    saved_batches = set()  # Use set for efficient 'in' operation

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            try:
                frame_id, frame = frame_save_queue.get(timeout=10)
                frame_batch.append((frame_id, frame))

                if len(frame_batch) >= batch_size or (frame_id // batch_size > current_batch_start // batch_size):
                    executor.submit(save_frame_worker, frame_batch[:], saved_batches, "./frames", 30, 3000)
                    frame_batch = []  # Clear the batch after submitting
                    current_batch_start = (frame_id // batch_size) * batch_size
            except queue.Empty:
                print("No new frames in queue for 10 seconds")
            except Exception as e:
                print(f"Error in frame saving thread: {e}")

def frame_capture_loop(rtsp_url, global_video_buffer, MAX_FRAMES_IN_BUFFER):
    while True:  # Outer loop to handle reconnection
        try:
            cap = cv2.VideoCapture(rtsp_url)
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    with threading.Lock():
                        if len(global_video_buffer) >= MAX_FRAMES_IN_BUFFER:
                            oldest_frame_id = next(iter(global_video_buffer))
                            del global_video_buffer[oldest_frame_id]
                        global_video_buffer[frame_id] = frame
                    frame_save_queue.put((frame_id, frame))
                    frame_id += 1
                else:
                    print("Lost connection to the camera. Attempting to reconnect...")
                    cap.release()  # Release the current connection
                    break  # Exit the inner loop to attempt reconnection

            # Optional: Add a delay before trying to reconnect
            time.sleep(5)  # Wait for 5 seconds before trying to reconnect

        finally:
            cap.release()

# Main loop for processing the video stream
def main_loop(global_video_buffer, lines_info, ip_address):
    last_processed_frame_id = None

    # GIL =? ASYCNIO => MULTIPR
    # GLOBAL INTERPRETRATOR LOCK
    while True:
        with threading.Lock():
            if global_video_buffer:
                frame_id, frame = next(reversed(global_video_buffer.items()))
                if frame_id == last_processed_frame_id:
                    # If it's the same, release the lock and wait
                    time.sleep(0.005)  # Adjust WAIT_TIME_FOR_NEW_FRAME as needed
                    continue 
                start_time = time.time()
                #window_name = "Processed Stream"
                #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                #cv2.resizeWindow(window_name, desired_size)
                #print(f"Adding frame {frame_id} to save queue")  # Debugging
                # frame_save_queue.put((frame_id, frame))

                try:
                    rois = [info[0] for info in lines_info if info[1] == "roi"]
                    processed_frame, detected_objects, tl_color = process_frame(frame, rois)
                    violation_detector.update_tracker(frame_id, detected_objects, ip_address)
                    line_positions = {}
                    for line in lines_info:
                            # Check if the line type is 'roi', which has a different structure
                            if line[1] == 'roi':
                                continue  # Skip the 'roi' line or handle it differently
                            else:
                                # For other lines, proceed as before
                                if line[2] not in line_positions:
                                    line_positions[line[2]] = []
                                line_positions[line[2]].append((line[0], line[1]))
                    
                    violation_detector.check_violations(frame_id, tl_color, line_positions, detected_objects)
                    
                    for line_info in lines_info:  # Convert stream_id to string to match key
                        if line_info[1] == "roi":
                            # Handle ROI which has four points
                            line_type = line_info[1]
                            points = line_info[0]
                            for i in range(len(points)):
                                cv2.line(processed_frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), colors[line_type], 2)
                        else:
                            # Handle other line types (1.1 line, stop_line, red_line) which have two points
                            start_point, end_point, line_type = line_info
                            cv2.line(processed_frame, tuple(start_point), tuple(end_point), colors[line_type], 2)
                    
                    for obj in detected_objects:
                        bbox = obj['bbox']  # bbox is in the format [x1, y1, x2, y2]
                        object_id = obj['id']
                        object_class = obj['class']

                        # Draw the bounding box
                        cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                        # Put text (ID and class) near the bounding box
                        text = f"ID: {object_id}, Class: {object_class}"
                        cv2.putText(processed_frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    #display_frame = cv2.resize(processed_frame, desired_size)
                    #cv2.imshow(window_name, display_frame)
                
                except Exception as e:
                    print(f"Error processing frame: {e}")

                end_time = time.time()
                fps = 1 / (end_time - start_time) if end_time > start_time else 0
                print(f"FPS: {fps:.2f}")

                last_processed_frame_id = frame_id

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    cv2.destroyAllWindows()


# ____________________________________________________________________________________________________
def get_rtsp_url(ip_address: str) -> str:
    return f"rtsp://admin:parol12345@{ip_address}/cam/realmonitor?channel=1&subtype=0"

def get_given_ip_rtsp_urls() -> list:
    ips: list = sys.argv[1:]

    return [
        get_rtsp_url(ip) for ip in ips
        ]
# ____________________________________________________________________________________________________


def main():
    urls = get_given_ip_rtsp_urls()

    

    violation_detector = ViolationDetector()
    global_video_buffer = OrderedDict()
    MAX_FRAMES_IN_BUFFER = 90
    lines_info = get_lines_info(rtsp_url, ip_address)


    device = torch.device("cpu")
    desired_size = (640, 640)

    capture_thread = threading.Thread(target=frame_capture_loop, args=(rtsp_url, global_video_buffer, MAX_FRAMES_IN_BUFFER))
    processing_thread = threading.Thread(target=main_loop, args=(global_video_buffer, lines_info, ip_address))
    saving_thread = threading.Thread(target=frame_saving_thread)
    
    # Start the capture and processing threads
    capture_thread.start()
    processing_thread.start()
    saving_thread.start()
    
    capture_thread.join()
    processing_thread.join()
    saving_thread.join()


if __name__ == "__main__":
    main()