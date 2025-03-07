"""Plottable objects
======================


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
    """This class defines several methods that should be common to
    all objects that contain data to be plotted"""

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

    def _process_2D_plot_args(self, ax, plane, limits, Npoints, cut):
        # ------------------------- START ARGUMENT CHECKING ----------------
        # - check plot config
        IMPLEMENTED_PLANES = ["XY", "YZ", "ZX"]
        if plane.upper() not in IMPLEMENTED_PLANES:
            raise ValueError(f"`plane` argument should be in {IMPLEMENTED_PLANES}")

        assert ax is None or isinstance(ax, Axes), "'ax' should be a matplotlib axis."
        # - check grid config
        # limits
        assert np.asanyarray(limits).size == 4, "`limits` should be an array of size 4"
        # Npoints
        Npoints = np.asanyarray(Npoints)
        msg = "`Npoints` should be an int or a list of three ints"
        assert Npoints.size in [1, 2], msg
        assert issubclass(Npoints.dtype.type, np.integer), msg
        # cut
        assert np.isscalar(cut), "'cut' should be a scalar"

        # ------------------------- STOP ARGUMENT CHECKING ----------------
        # - init ax (if needed)
        ax = self._init_ax(ax)

        # - init meshgrid
        xmin, xmax, ymin, ymax = limits
        Nx, Ny = (Npoints, Npoints) if Npoints.size == 1 else Npoints
        # depending on plane
        match plane.upper():
            case "XY":
                grid = np.mgrid[
                    xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j, cut:cut:1j
                ]
                position = grid.T[0]
                X, Y, _ = position.T
            case "YZ":
                grid = np.mgrid[
                    cut:cut:1j, xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j
                ]
                position = grid.T
                position = np.moveaxis(position, 2, 0)
                position = position[0]
                _, X, Y = position.T

            case "ZX":
                grid = np.mgrid[
                    ymin : ymax : Ny * 1j, cut:cut:1j, xmin : xmax : Nx * 1j
                ]
                position = grid.T
                position = np.moveaxis(position, 1, 0)
                position = position[0]
                Y, _, X = position.T

        return ax, position, X.T, Y.T
