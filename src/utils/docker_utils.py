"""
docker_utils.py

Utilities for Docker environment.
"""

import os


def is_running_in_docker():

    return os.path.exists(
        "/.dockerenv"
    )


def get_environment():

    if is_running_in_docker():

        return "docker"

    return "local"


if __name__ == "__main__":

    print(
        f"Environment: {get_environment()}"
    )