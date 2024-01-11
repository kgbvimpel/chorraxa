import asyncio
from utils import get_camera_ips
from processes import capture_frame_process_1
from multiprocessing import Process


# 12 threads, 6 core

async def main():
    for ip, url in get_camera_ips():
        await capture_frame_process_1(ip, url=url)


if __name__ == "__main__":
    asyncio.run(main())
