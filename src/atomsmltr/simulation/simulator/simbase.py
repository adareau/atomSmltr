"""simulators objects
=======================

Here we implements the generic ``Simulation`` class. Actual implementions are to be
defined in other submodules.
"""

# % IMPORTS
import numpy as np
from scipy import constants as csts
from abc import ABC, abstractmethod
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass

# % LOCAL IMPORTS
from ..configurator import Configuration

# % USEFUL FUNCTIONS


def _get_force_vec(
    position: np.ndarray,
    speed: np.ndarray,
    config: Configuration,
    return_list: bool = False,
) -> np.ndarray:
    """Computes the force on an atom, by adding all radiations pressures,
    in a vectorization style that matches the package's standards

    Parameters
    ----------
    position : array, shape (3,) or (n1, n2, .., 3)
        array of cartesian positions in the lab frame
    speed : array, shape (3,) or (n1, n2, .., 3)
        array of cartesian speeds in the lab frame
    config : Configuration
        a configuration object
    return_list : bool (opt, default=False)
        if set to True, also returns the scattering rates and laser wavenumbers
        this is used to compute the stochastic part of the force (spont. em.)

    Returns
    -------
    force : array, (3,) or (n1, n2, .., 3)
        the force felt by the atoms

    scattering_list : list
        returned only if ``return_list`` is set to ``True``
        list of lasers scattering rates and wavenumbers.
    """

    # - get magnetic field value & norm
    B = config.getB(position)
    Bx, By, Bz = B.T
    B_norm = np.sqrt(Bx**2 + By**2 + Bz**2).T
    # - initialize force
    force = np.zeros_like(position, dtype=float)
    # - prepare scattering list
    scattering_list = []
    # - loop over atom-light couplings
    atomlight_couples = config.get_atomlight_couples()
    for elements in atomlight_couples:
        transition, laser, detuning = elements
        laser_intensity = laser.get_value(position)
        polarization = laser.get_polarization_quant(B)
        # Doppler
        det_Doppler = -np.dot(speed, transition.k * laser.unit_vector)
        scattering_rate = transition.get_scattering_rate(
            laser_intensity, B_norm, polarization, detuning + det_Doppler
        )
        radiation_pressure = csts.hbar * transition.k * scattering_rate
        force = force + radiation_pressure[..., np.newaxis] * laser.unit_vector
        if return_list:
            scattering_list.append(
                {
                    "rate": scattering_rate,
                    "k": transition.k,
                    "unit_vector": laser.unit_vector,
                }
            )

    # - loop over all forces
    for f in config.get_all_forces():
        force = force + f.get_value(position)

    if return_list:
        return force, scattering_list
    else:
        return force


def get_force_vec(
    pos_speed_vector: np.ndarray,
    config: Configuration,
    return_list: bool = False,
) -> np.ndarray:
    """Computes the force on an atom, by adding all radiations pressures,
    in a vectorization style that matches the package's standards

    Parameters
    ----------
    pos_speed_vector : array, shape (6,) or (n1, n2, .., 6)
        array of cartesian coordinates (position and speed) in the lab frame
    config : Configuration
        a configuration object
    return_list : bool (opt, default=False)
        if set to True, also returns the scattering rates and laser wavenumbers
        this is used to compute the stochastic part of the force (spont. em.)

    Returns
    -------
    force : array, shape (3,) or (n1, n2, .., 3)
        the force at the coordinates given by ``pos_speed_vector``

    scattering_list : list
        returned only if ``return_list`` is set to ``True``
        list of lasers scattering rates and wavenumbers.

    Notes
    -----
    ``pos_speed_vector`` is an array_like object, with shape (6,) or (n1, n2, .., 6).

    In all cases, the last dimension contains cordinates (x, y, z, vx, vy, vz),
    in meter or meter/seconds and in the lab frame

    Examples
    --------

    .. code-block:: python

        # ... init a config object with the `Configuration` class
        from atomsmltr.simulation.simulator import get_force_vec
        import numpy as np

        # - init a position & speed vector grid
        # x spans from -0.1 to 0.1
        # vx spans from -10 to 30
        # y, z, vy, vz set to 0
        grid = np.mgrid[
            -0.1:0.1:100j,  # x
                0:0:1j,  # y
                0:0:1j,  # z
            -10:30:101j,  # vx
                0:0:1j,  # vy
                0:0:1j,  # vz
        ]
        # squeeze unused dimensions
        grid = np.squeeze(grid)
        # get X and VX grids (for instance for plotting)
        X, _, _, VX, _, _ = grid
        # make (x, y, z, vx, vy, vz) the last dimension
        # as requested by vectorization convention
        pos_speed_vector = grid.T

        # - compute the force
        force = get_force_vec(pos_speed_vector, config)
        FX, FY, FZ = force.T

        # - print shapes for illustration
        print(f"{grid.shape=}")
        print(f"{X.shape=}")
        print(f"{FX.shape=}")
        print(f"{pos_speed_vector.shape=}")
        print(f"{force.shape=}")


    This returns

    .. code-block:: python

        grid.shape=(6, 100, 101)
        X.shape=(100, 101)
        FX.shape=(100, 101)
        pos_speed_vector.shape=(101, 100, 6)
        force.shape=(101, 100, 3)
    """

    # TODO should we move that to the Configuration class ???
    # - get position and speed
    x, y, z, vx, vy, vz = pos_speed_vector.T
    position = np.array([x, y, z]).T
    speed = np.array([vx, vy, vz]).T
    # - get force
    res = _get_force_vec(position, speed, config, return_list)
    if return_list:
        force, scattering_list = res
        return force, scattering_list
    else:
        force = res
        return force


# % DEFINE THE BASE CLASS


@dataclass
class SimRes:
    """Class for simulation results"""

    y: np.ndarray
    t: np.ndarray
    y_last: np.ndarray = None
    tags: set = None
    t_events: list = None
    y_events: list = None
    success: bool = True


class Simulation(ABC):
    """The generic Simulation object

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider, by default None

    Note
    -----
        this is an abstract class, actual implementations are
        defined elsewhere and inherit from this class
    """

    def __init__(self, config: Configuration = None):
        super(Simulation, self).__init__()
        if config is not None:
            self.config = config
        self.u0_list = []

    # -- SETTERS AND GETTERS
    @property
    def config(self):
        """Configuration: the configuration for this simulation"""
        return self.__config

    @config.setter
    def config(self, value):
        if not isinstance(value, Configuration):
            raise TypeError("'config' should be a `Configuration` object.")
        self.__config = value

    @property
    def u0_list(self):
        """list: a list of initial conditions for batch running"""
        return self.__u0_list

    @u0_list.setter
    def u0_list(self, value):
        value = self._u0_list_checker(value)
        self.__u0_list = value

    # -- REQUESTED FUNCTIONS
    @abstractmethod
    def get_force(self, u: np.ndarray) -> np.ndarray:
        """returns the force felt at a position/speed vector u

        Parameters
        ----------
        u : array, shape (6,) or (n1, n2, .., 6)
            array of cartesian coordinates (position and speed) in the lab frame

        Returns
        -------
        force : array, shape (3,) or (n1, n2, .., 3)
            the force at the coordinates given by ``pos_speed_vector``
        """
        pass

    def integrate(self, u0: np.ndarray, t: np.ndarray):
        """Integrates the system with initial conditions ``u0``

        Parameters
        ----------
        u0 : array, shape (6,)
            the initial conditions (x, y, z, vx, vy, vz)
        t : array, shape (n,)
            the timesteps to integrate

        Returns
        -------
        res
            the result of the simulation
        """
        ####################
        #  PRE PROCESSING  #
        ####################
        # for later use

        #################
        #  INTEGRATION  #
        #################
        #  integrate using the method specific `_integrate()` method
        res = self._integrate(u0, t)

        #####################
        #  POST PROCESSING  #
        #####################
        # - apply zones tags
        # get zones
        position_zones, speed_zones = self.config.get_all_zones()
        # get last position
        # -------------------------------------------------------
        # Note : we have to take into account the case where
        #        u0 is a vector of shape (n, m, ..., 6)
        #        and stop times might be different for all
        #        dimensions. In this case, when one trajectory
        #        is "stopped", it is filled with nan. Thus we
        #        will take all values for u backwards in time,
        #        and replace all nans until we have no nans
        # -------------------------------------------------------
        # 1) take last value
        # since res.y has a shape (n, m, ..., 6, N) where
        # N is the number of timesteps, we transpose to make
        # it easier to iterate on timesteps
        yT = res.y.T
        # take the last time step
        uT_last = yT[-1, ...]
        # iterate backward in time
        for uT in yT[::-1]:
            # we replace the nan values in the current vector
            # by the ones from the last timestep on which we iterate
            # non nan values are kept
            uT_last = np.where(np.isnan(uT_last), uT, uT_last)
            # if we have no nan left, we stop
            if not np.any(np.isnan(uT_last)):
                break
        # transpose it back
        u_last = uT_last.T
        # store it
        res.y_last = u_last
        # extract speed and position
        position = u_last[..., :3]
        speed = u_last[..., 3:]
        res.tags = set()  # we use a set to have unique values
        # add position tags
        for zone in position_zones:
            new_tags = np.where(
                zone.get_value(position),
                {zone.in_tag},
                {zone.out_tag},
            )
            res.tags |= new_tags
        # add speed tags
        for zone in speed_zones:
            new_tags = np.where(
                zone.get_value(speed),
                {zone.in_tag},
                {zone.out_tag},
            )
            res.tags |= new_tags

        return res

    @abstractmethod
    def _integrate(self, u0, t):
        """actual integration"""
        pass

    @abstractmethod
    def dudt(self, t, u):
        """should return the derivative of the position/speed vector u"""
        pass

    @abstractmethod
    def _u0_list_checker(self, value):
        """checks that the list of initial conditions matches what is expected
        for a given simulator implementation"""
        pass

    # -- RUN
    def run(
        self,
        t: np.ndarray,
        u0_list: list = None,
        npools: int = 0,
        verbose: bool = False,
    ) -> list:
        """Runs a batch of simulations from a list of initial conditions

        Parameters
        ----------
        t : array, shape (n,)
            time steps for the simulation
        u0_list : list, optional
            list of initial conditions, by default None
        npools : int, optional
            number of pools for parallel computing.
            If set to zero, no paralalelisation, by default 0
        verbose : bool, optional
            if set to True, a progress bar is displayed, by default False

        Returns
        -------
        res_list : list
            a list of results

        Examples
        --------

        .. code-block:: python

            # ... init a config object with the `Configuration` class

            # - import a simulation class
            from atomsmltr.simulation import ScipyIVP_3D

            # - init and setup
            sim = ScipyIVP_3D(method="Radau")
            sim.config = config

            # - parameters
            # initial conditions
            vz_list = np.linspace(10, 300, 40)
            u0_list = [(0, 0, -0.15, 0, 0, v) for v in vz_list]
            sim.u0_list = u0_list
            # time
            t = np.linspace(0, 0.05, 1000)

            # - run a batch in parallel
            res_list = sim.run(t, npools=5, verbose=True)

        """
        if u0_list is not None:
            self.u0_list = u0_list
        if not isinstance(npools, int):
            return TypeError("'npools' should be an int")
        if npools:
            map_fun = partial(self.integrate, t=t)
            if verbose:
                Nmax = len(self.u0_list)
                res_list = []
                with Pool(npools) as p, tqdm(total=Nmax) as pbar:
                    for res in p.imap(map_fun, self.u0_list):
                        pbar.update()
                        pbar.refresh()
                        res_list.append(res)
            else:
                with Pool(npools) as p:
                    res_list = p.map(map_fun, self.u0_list)
        else:
            res_list = []
            u0_list = tqdm(self.u0_list) if verbose else self.u0_list
            for u0 in u0_list:
                res = self.integrate(u0, t)
                res_list.append(res)
        return res_list
