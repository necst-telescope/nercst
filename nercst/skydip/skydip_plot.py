import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
import numpy as np
import re

from .skydip import Skydip
from ..core import io


def calc_figsize(topicname_list: list):
    figsize_x = np.round(np.sqrt(len(topicname_list))).astype(int)
    figsize_y = int(len(topicname_list) // figsize_x)
    if len(topicname_list) % figsize_x > 0:
        figsize_y += 1
    while len(topicname_list) < figsize_x * figsize_y:
        topicname_list.append(None)
    return figsize_x, figsize_y, topicname_list


def plot_all(
    dbname: Path,
    telescop: Literal["NANTEN2", "OMU1p85m", "previous"],
    save=False,
):
    """
    Plot results for all topic names.

    Parameters
    ----------
    dbname: Path
        Path to the database directory
    telescop
        Name of telescope
    save: bool
        "True" -> save this figure named as "..._skydip.pdf" in dbname.parent directory.

    Examples
    --------
    >>> skydip.plot_all(dbname)
    (Show results for all topic names.)
    """
    board_list = sorted(io.board_name_getter(dbname))
    figsize_x, figsize_y, board_list = calc_figsize(board_list)
    fig, ax = plt.subplots(
        figsize_x, figsize_y, figsize=(5 * figsize_x + 3, 5 * figsize_y)
    )
    for i, boad_name in enumerate(board_list):
        if boad_name is not None:
            print(f"calc {boad_name}...")
            db = io.loaddb(dbname, boad_name, telescop)
            skydip = Skydip(db)
            _ = skydip.plot(
                ax[i // figsize_y, i % figsize_x],
                re.search(r"board\d", boad_name).group(),
            )
    fig.suptitle(dbname.stem)
    fig.tight_layout()
    if save:
        if "skydip" in str(dbname).lower():
            fig.savefig(dbname.with_suffix(".pdf"))
        else:
            fig.savefig(dbname.parent.joinpath(str(dbname.name) + "_skydip.pdf"))
