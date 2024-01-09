from sys import argv as Argvs

def get_camera_rtsp(ip: str, user: str = "admin", password: str = "parol12345") -> str:
    return f"rtsp://{user}:{password}@{ip}/cam/realmonitor?channel=1&subtype=0"

def get_camera_ips() -> tuple[str]:
    return (
        (ip, get_camera_rtsp(ip)) for ip in Argvs[1:]
    )