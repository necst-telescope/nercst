import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
import numpy as np

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


def plot_all(dbname: Path, telescop: Literal["NANTEN2", "OPU1.85"]):
    """
    Plot results for all topic names.

    Parameters
    ----------
    dbname: Path
        Path to the database directory
    telescop
        Name of telescope

    Examples
    --------
    >>> rsky.plot_all(dbname,"NANTEN2")
    (Show results for all topic names.)
    """
    topicname_list = io.topic_getter(dbname)
    figsize_x, figsize_y, topicname_list = calc_figsize(topicname_list)
    fig, ax = plt.subplots(
        figsize_x, figsize_y, figsize=(5 * figsize_x + 3, 5 * figsize_y)
    )
    for i, topicname in enumerate(topicname_list):
        if topicname is not None:
            db = io.loaddb(dbname, topicname, telescop)
            r_sky = Rsky(db)
            r_sky.tsys()
            r_sky.plot(fig, ax[i // figsize_y, i % figsize_x])
    plt.tight_layout()
