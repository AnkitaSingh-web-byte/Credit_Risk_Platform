"""
helpers.py

Common helper functions.
"""

import json
from pathlib import Path


def save_json(
    data,
    file_path
):

    file_path = Path(file_path)

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


def load_json(
    file_path
):

    file_path = Path(file_path)

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def format_currency(
    value
):

    return f"₹{value:,.2f}"


def format_percentage(
    value
):

    return f"{value:.2%}"