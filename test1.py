import asyncio


async def task1():
    print("Task 1 started")
    await asyncio.sleep(2)
    print("Task 1 completed")


async def task2():
    print("Task 2 started")
    await asyncio.sleep(1)
    print("Task 2 completed")


async def task3():
    print("Task 3 started")
    await asyncio.sleep(3)
    print("Task 3 completed")


async def main():
    # Run tasks concurrently using asyncio.gather
    await asyncio.gather(
        task1(),
        task2(),
        task3(),
    )

if __name__ == "__main__":
    asyncio.run(main())
