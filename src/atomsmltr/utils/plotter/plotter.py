# -*- coding: utf-8 -*-
"""Defines the plotter classes
"""

# % IMPORTS
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes

# % LOCAL IMPORTS
from .tools import Axes3D


# % ABSTRACT CLASS


class Plottable(ABC):
    """A class to describe plottable objects"""

    def __init__(self):
        super(Plottable, self).__init__()

    def _init_ax(self, ax=None, ax3D=False):
        if ax is None:
            fig = plt.figure()
            if ax3D:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)
        return ax

    @abstractmethod
    def plot1D(self, ax=None):
        pass

    @abstractmethod
    def plot2D(self, ax=None, plane="XY"):
        pass

    @abstractmethod
    def plot3D(self, ax=None):
        pass

    def _process_2D_plot_args(self, ax, plane, limits, Npoints, X, Y):
        # ------------------------- START ARGUMENT CHECKING ----------------
        # - check plot config
        IMPLEMENTED_PLANES = ["XY", "YZ", "ZX"]
        if plane.upper() not in IMPLEMENTED_PLANES:
            raise ValueError(f"`plane` argument should be in {IMPLEMENTED_PLANES}")

        assert ax is None or isinstance(ax, Axes), "'ax' should be a matplotlib axis."
        # - check axis config
        # general
        if (limits is not None) + (Npoints is not None) + (X is not None) + (
            Y is not None
        ) > 2:
            msg = "Too many arguments given for meshgrid definition. "
            msg += "Either provide `limits` and `Npoints` or `X` and `Y`"
            raise ValueError(msg)
        if (limits is None) + (Npoints is None) == 1:
            raise ValueError("Both `limits` and `Npoints` arguments have to be passed")
        if (X is None) + (Y is None) == 1:
            raise ValueError("You have to provide both `X` and `Y` mesh")
        # argument per argument
        # limits
        assert (
            limits is None or np.asanyarray(limits).size == 4
        ), "`limits` should be an array of size 4"
        # Npoints
        if Npoints is not None:
            Npoints = np.asanyarray(Npoints)
            msg = "`Npoints` should be an int or a list of two ints"
            assert Npoints.size in [1, 2], msg
            assert issubclass(Npoints.dtype.type, np.integer), msg
        # X, Y
        if X is not None:
            X = np.asanyarray(X)
            Y = np.asanyarray(Y)
            assert X.shape == Y.shape, "'X' and 'Y' should have the same shape"

        # ------------------------- STOP ARGUMENT CHECKING ----------------
        # - init ax (if needed)
        ax = self._init_ax(ax)

        # - init meshgrid
        if X is None:
            xmin, xmax, ymin, ymax = limits
            Nx, Ny = (Npoints, Npoints) if Npoints.size == 1 else Npoints
            print(Nx)
            x = np.linspace(xmin, xmax, Nx)
            y = np.linspace(ymin, ymax, Ny)
            X, Y = np.meshgrid(x, y)

        return ax, X, Y
