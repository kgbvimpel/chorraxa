from dataclasses import dataclass
from os import path as OSPath, mkdir
import argparse
import configparser
from models import CameraIP


def create_folders(foldername: str, ips: list[str]) -> list[str]:
    _path = OSPath.join(OSPath.dirname(__file__), foldername)

    if not OSPath.exists(_path):
        mkdir(_path)

    folders = []
    for ip in ips:
        camera_foldername = OSPath.join(_path, ip)
        if not OSPath.exists(camera_foldername):
            mkdir(camera_foldername)
        folders.append(camera_foldername)
    return folders


def camera_ips() -> tuple[CameraIP]:
    parser = argparse.ArgumentParser(
        description='Example script with argparse and config file.')

    # Add the -c/--config parameter
    parser.add_argument('-c', '--config', default='config.ini',
                        help='Specify the config file path')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    # Access values from the configuration file
    username = config.get('Settings', 'username').strip()
    password = config.get('Settings', 'password').strip()
    ips = [ip.strip() for ip in config.get('Settings', 'ips').split(',')]

    # Get foldername
    foldername = config.get('Main', 'foldername').strip()
    folders = create_folders(foldername=foldername, ips=ips)

    return (
        CameraIP(
            ip=ip,
            url="rtsp://{}:{}@{}/cam/realmonitor?channel=1&subtype=0".format(
                username, password, ip),
            folder=folder
        ) for ip, folder in zip(ips, folders)
    )
