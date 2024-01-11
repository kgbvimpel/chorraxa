from models import Frame
from time import sleep
import asyncio


async def func(frame: Frame, i):
    await asyncio.sleep(3)
    frame.set_last_frame(i)
    await asyncio.sleep(1)


async def task1():
    frame = Frame()

    await asyncio.gather(
        func(frame, '192.168.1.1'),
        func(frame, '192.168.1.2'),
    )

    print('Task1:', frame.get_last_frames())

    return 'Task2 done!'


async def task3():
    frame = Frame()

    await asyncio.sleep(4)
    frame.set_last_frame('192.168.1.1', [1, 2, 3])


async def task2():
    frame = Frame()

    await asyncio.sleep(4)

    print('Task2:', frame.get_last_frames())

    await asyncio.sleep(2)

    return 'Task2 done!'


async def main():
    results = await asyncio.gather(
        task1(),
        task2(),
        task3(),
    )

    for result in results:
        print(result)


if __name__ == "__main__":
    import utils

    print(utils.frames_folder())
    asyncio.run(main())
