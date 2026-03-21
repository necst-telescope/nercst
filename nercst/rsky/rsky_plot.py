import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
import numpy as np
import re

from .rsky import Rsky
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
    telescop: Literal["NANTEN2", "OPU1.85", "Common"] = "Common",
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
        "True" -> save this figure named as "..._rsky.pdf" in dbname.parent directory.

    Examples
    --------
    >>> rsky.plot_all(dbname)
    (Show results for all topic names.)
    """
    if type(dbname) == str:
        dbname = Path(dbname)
    board_list = sorted(
        io.board_name_getter(dbname), key=lambda x: int(x.split("board")[-1])
    )
    figsize_x, figsize_y, board_list = calc_figsize(board_list)
    fig, ax = plt.subplots(
        figsize_x, figsize_y, figsize=(5 * figsize_x + 3, 5 * figsize_y)
    )
    for i, boad_name in enumerate(board_list):
        if boad_name is not None:
            db = io.loaddb(dbname, boad_name, telescop)
            r_sky = Rsky(db)
            r_sky.tsys()
            r_sky.plot(
                fig,
                ax[i // figsize_y, i % figsize_x],
                re.search(r"board\d", boad_name).group(),
            )
    plt.tight_layout()
    if save:
        if "rsky" in str(dbname).lower():
            fig.savefig(dbname.with_suffix(".pdf"))
        else:
            fig.savefig(dbname.parent.joinpath(str(dbname.name) + "_rsky.pdf"))
