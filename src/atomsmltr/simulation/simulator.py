"""simulator
==================

Here we implement the ``Simulation`` class, that allows to run simulations
on a configuration defined via the ``Configuration`` class.

Examples
--------------------

Run a simulation with one initial condition vector

.. code-block:: python

        # ... init a config object with the `Configuration` class

        # - import a simulation class
        from atomsmltr.simulation import ScipyIVP_3D

        # - init and setup
        sim = ScipyIVP_3D(method="Radau")
        sim.config = config

        # - parameters
        u0 = (0, 0, -0.15, 0, 0, 200)
        t = np.linspace(0, 0.05, 1000)

        # - integrate
        res = sim.integrate(u0, t)

Run a batch of simulations

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

# % IMPORTS
import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants as csts
from abc import ABC, abstractmethod
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

# % LOCAL IMPORTS
from .configurator import Configuration

# % USEFUL FUNCTIONS


def get_force_vec_scipy(
    pos_speed_vector: np.ndarray, config: Configuration
) -> np.ndarray:
    """Computes the force on an atom, by adding all radiations pressures,
    in a Scipy compatible vectorization style

    Parameters
    ----------
    pos_speed_vector : array, shape (6,) or (6,k)
        cartesian position and speed vector
    config : Configuration
        a configuration object

    Returns
    -------
    force : array, shape (3,) or (3,k)
        the force at the coordinates given by ``pos_speed_vector``

    Notes
    -----

    The position/speed vector 'pos_speed_vector' should be of shape (6,) or (6,k)
    with the first dimension containing (x, y, z, vx, vy, vz) in the lab frame

    Note
    ----
        The function is vectorized to be compatible with Scipy's ``solve_ivp``
        function. Hence, it does not satisfy the functionnal vectorization
        used in the rest of this module

    Examples
    ---------

    .. code-block:: python

        # ... init a config object with the `Configuration` class
        from atomsmltr.simulation.simulator import get_force_vec_scipy
        import numpy as np

        # - init a position & speed vector grid
        # vx spans from -10 to 30
        # x, y, z, vy, vz set to 0
        vx_list = np.linspace(-10, 30, 301)
        pos_speed_vector = np.array([(0,0,0,vx,0,0,) for vx in vx_list]).T

        # - compute the force
        force = get_force_vec_scipy(pos_speed_vector, config)
        FX, FY, FZ = force

        # - print shapes for illustration
        print(f"{FX.shape=}")
        print(f"{pos_speed_vector.shape=}")
        print(f"{force.shape=}")


    This returns

    .. code-block:: python

        FX.shape=(301,)
        pos_speed_vector.shape=(6, 301)
        force.shape=(3, 301)


    """
    # TODO should we move that to the Configuration class ???
    # - get position and speed
    position = pos_speed_vector[0:3, ...].T
    speed = pos_speed_vector[3:6, ...].T
    # - get magnetic field value & norm
    B = config.getB(position)
    Bx, By, Bz = B.T
    B_norm = np.sqrt(Bx**2 + By**2 + Bz**2).T
    # - initialize force
    force = np.zeros_like(position, dtype=float)
    # - loop over atom-light couplings
    atomlight_couples = config.get_atomlight_couples()
    for elements in atomlight_couples:
        transition, laser, detuning = elements
        laser_intensity = laser.get_intensity(position)
        polarization = laser.get_polarization_quant(B)
        # Doppler
        det_Doppler = -np.dot(speed, laser.kvec)
        scattering_rate = transition.get_scattering_rate(
            laser_intensity, B_norm, polarization, detuning + det_Doppler
        )
        radiation_pressure = csts.hbar * transition.k * scattering_rate
        force = force + radiation_pressure[..., np.newaxis] * laser.unit_vector

    return force.T


def get_force_vec(pos_speed_vector: np.ndarray, config: Configuration) -> np.ndarray:
    """Computes the force on an atom, by adding all radiations pressures,
    in a vectorization style that matches the package's standards

    Parameters
    ----------
    pos_speed_vector : array, shape (6,) or (n1, n2, .., 6)
        array of cartesian coordinates (position and speed) in the lab frame
    config : Configuration
        a configuration object

    Returns
    -------
    array, shape (3,) or (n1, n2, .., 3)
        the force at the coordinates given by ``pos_speed_vector``

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
    # - get magnetic field value & norm
    B = config.getB(position)
    Bx, By, Bz = B.T
    B_norm = np.sqrt(Bx**2 + By**2 + Bz**2).T
    # - initialize force
    force = np.zeros_like(position, dtype=float)
    # - loop over atom-light couplings
    atomlight_couples = config.get_atomlight_couples()
    for elements in atomlight_couples:
        transition, laser, detuning = elements
        laser_intensity = laser.get_intensity(position)
        polarization = laser.get_polarization_quant(B)
        # Doppler
        det_Doppler = -np.dot(speed, laser.kvec)
        scattering_rate = transition.get_scattering_rate(
            laser_intensity, B_norm, polarization, detuning + det_Doppler
        )
        radiation_pressure = csts.hbar * transition.k * scattering_rate
        force = force + radiation_pressure[..., np.newaxis] * laser.unit_vector

    return force


# % DEFINE THE BASE CLASS


class Simulation(ABC):
    """docstring for Simulator."""

    def __init__(self, config=None):
        super(Simulation, self).__init__()
        if config is not None:
            self.config = config
        self.u0_list = []

    # -- SETTERS AND GETTERS
    @property
    def config(self):
        return self.__config

    @config.setter
    def config(self, value):
        if not isinstance(value, Configuration):
            raise TypeError("'config' should be a `Configuration` object.")
        self.__config = value

    @property
    def u0_list(self):
        return self.__u0_list

    @u0_list.setter
    def u0_list(self, value):
        value = self._u0_list_checker(value)
        self.__u0_list = value

    # -- REQUESTED FUNCTIONS
    @abstractmethod
    def get_force(self, u):
        pass

    def integrate(self, u0, t):
        """Integrates the system

        Args:
            u0 (array): initial conditions
            t (array): timesteps for the integration

        Returns:
            res: result of the integration (might depend on the method used)

        """
        return self._integrate(u0, t)

    @abstractmethod
    def _integrate(self, u0, t):
        """actual integration"""
        pass

    @abstractmethod
    def dudt(self, t, u):
        pass

    @abstractmethod
    def _u0_list_checker(self, value):
        pass

    # -- RUN
    def run(self, t, u0_list=None, npools=0, verbose=False):
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


# % SIMULATOR BASED ON SCIPY'S SOLVE_IVP


def stop_position_event(t, u, stop_position):
    position = u[0:3, ...].T
    res = np.logical_and.reduce([zone.in_zone(position) for zone in stop_position])
    return res


def stop_speed_event(t, u, stop_speed):
    speed = u[3:6, ...].T
    res = np.logical_and.reduce([zone.in_zone(speed) for zone in stop_speed])
    return res


class ScipyIVP_3D(Simulation):
    """docstring for ScipyIVP_3D."""

    def __init__(self, config=None, method="Radau", **solve_ivp_args):
        super(ScipyIVP_3D, self).__init__(config)
        self.solve_ivp_args = solve_ivp_args
        self.method = method

    # -- REQUESTED FUNCTIONS
    def _get_force_scipy(self, u):
        force = get_force_vec_scipy(u, self.config)
        return force

    def get_force(self, u):
        force = get_force_vec(u, self.config)
        return force

    def dudt(self, t, u):
        F = self._get_force_scipy(u)
        _, _, _, vx, vy, vz = u
        dx, dy, dz = vx, vy, vz
        dvx, dvy, dvz = F / self.config.atom.mass
        res = np.array([dx, dy, dz, dvx, dvy, dvz])
        return res

    def _integrate(self, u0, t):
        # - u0 to array
        u0 = np.asanyarray(u0)
        # - get stop events
        events = []
        stop_position, stop_speed = self.config.get_stop_zones()
        if stop_position:
            stop_pos = partial(stop_position_event, stop_position=stop_position)
            stop_pos.terminal = True
            events.append(stop_pos)
        if stop_speed:
            stop_sp = partial(stop_speed_event, stop_speed=stop_speed)
            stop_sp.terminal = True
            events.append(stop_sp)
        # - time
        t = np.asanyarray(t)
        if not t.shape:
            t = np.asanyarray([0, t])
        t = np.sort(t)
        t_span = (t[0], t[-1])
        # - integrate
        res = solve_ivp(
            fun=self.dudt,
            t_span=t_span,
            y0=u0,
            method=self.method,
            t_eval=t,
            events=events,
            **self.solve_ivp_args,
        )
        return res

    def _stop_event_speed(self, t, u):
        pass

    def _stop_event_position(self, t, u):
        pass

    def _u0_list_checker(self, value):
        if not hasattr(value, "__iter__"):
            raise ValueError("'u0_list' should be an iterable object")
        if value:
            for u0 in value:
                if np.asanyarray(u0).shape != (6,):
                    raise ValueError("'u0_list' should be a list of arrays of size 6")
        return value
