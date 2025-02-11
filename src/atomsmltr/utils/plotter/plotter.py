# -*- coding: utf-8 -*-
"""Defines the plotter classes
"""

# % IMPORTS
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# % LOCAL IMPORTS
from .tools import Axes3D


# % ABSTRACT CLASS


class Plottable(ABC):
    """A class to describe plottable objects"""

    def __init__(self):
        pass

    def _init_ax(self, ax=None, ax3D=False):
        if ax is None:
            fig = plt.figure()
            if ax3D:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        return ax

    @abstractmethod
    def plot1D():
        pass

    @abstractmethod
    def plot2D():
        pass

    @abstractmethod
    def plot3D():
        pass
