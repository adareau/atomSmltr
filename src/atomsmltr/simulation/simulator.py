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
from dataclasses import dataclass

# % LOCAL IMPORTS
from .configurator import Configuration

# % USEFUL FUNCTIONS


def _get_force_vec(
    position: np.ndarray, speed: np.ndarray, config: Configuration
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

    Returns
    -------
    force : array, (3,) or (n1, n2, .., 3)
        the force felt by the atoms
    """

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

    # - loop over all forces
    for f in config.get_all_forces():
        force = force + f.value(position)

    return force


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
    # - compute force
    force = _get_force_vec(position, speed, config)
    # - transpose to satisfy vectorization rules
    force = force.T
    return force


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
    force : array, shape (3,) or (n1, n2, .., 3)
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
    # - get force
    force = _get_force_vec(position, speed, config)
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
                zone.in_zone(position),
                {zone.in_tag},
                {zone.out_tag},
            )
            res.tags |= new_tags
        # add speed tags
        for zone in speed_zones:
            new_tags = np.where(
                zone.in_zone(speed),
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


# % SIMULATOR BASED ON SCIPY'S SOLVE_IVP


def stop_position_event_scipy(
    t: float, u: np.ndarray, stop_position: list, offset: float = 0.0
):
    """Implements 'stop' events for Scipy's solve_ivp, based on atom's position

    Parameters
    ----------
    t : float
        time, not used here but required for the ``events`` functions in ``solve_ivp``
    u : array, shape (6,k)
        position/speed vector, according to ``solve_ivp`` vectorization convention
    stop_position : list
        list of Zones objects targetting position with actions set to stop

    Returns
    -------
    res: bool
        whether to stop the simulation

    See also
    --------
    atomsmltr.environment.zones
    atomsmltr.simulation.configurator.Configuration.get_stop_zones()
    """
    position = u[0:3, ...].T
    res = np.logical_and.reduce([zone.in_zone(position) for zone in stop_position])
    res = res + offset
    return res


def stop_speed_event_scipy(
    t: float, u: np.ndarray, stop_speed: list, offset: float = 0.0
):
    """Implements 'stop' events for Scipy's solve_ivp, based on atom's speed

    Parameters
    ----------
    t : float
        time, not used here but required for the ``events`` functions in ``solve_ivp``
    u : array, shape (6,k)
        position/speed vector, according to ``solve_ivp`` vectorization convention
    stop_speed : list
        list of Zones objects targetting speed with actions set to stop

    Returns
    -------
    res: bool
        whether to stop the simulation

    See also
    --------
    atomsmltr.environment.zones
    atomsmltr.simulation.configurator.Configuration.get_stop_zones()
    """
    speed = u[3:6, ...].T
    res = np.logical_and.reduce([zone.in_zone(speed) for zone in stop_speed])
    res = res + offset
    return res


class ScipyIVP_3D(Simulation):
    """A simulation class based on Scipy's ``solve_ivp`` solver

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider for the simulation
    method : str, optional
        method used for the ``solve_ivp`` solver, by default "Radau"
    **solve_ivp_args
        all other arguments are directly passed to ``solve_ivp``

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

    """

    def __init__(
        self, config: Configuration = None, method: str = "Radau", **solve_ivp_args
    ):
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
            stop_pos = partial(
                stop_position_event_scipy, stop_position=stop_position, offset=-0.5
            )
            stop_pos.terminal = True
            events.append(stop_pos)
        if stop_speed:
            stop_sp = partial(
                stop_speed_event_scipy, stop_speed=stop_speed, offset=-0.5
            )
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


# % HOME-MADE SIMULATORS


def stop_position_event(u: np.ndarray, stop_position: list):
    """Implements 'stop' events for home-made simulators, based on atom's position

    Parameters
    ----------
    u : array, shape (n,m,...,6)
        position/speed vector, according to our vectorization convention
    stop_position : list
        list of Zones objects targetting position with actions set to stop

    Returns
    -------
    res: bool
        whether to stop the simulation

    See also
    --------
    atomsmltr.environment.zones
    atomsmltr.simulation.configurator.Configuration.get_stop_zones()
    """
    x, y, z, _, _, _ = u.T
    position = np.array([x, y, z]).T
    res = np.logical_and.reduce([zone.in_zone(position) for zone in stop_position])
    res = res
    return res


def stop_speed_event(u: np.ndarray, stop_speed: list):
    """Implements 'stop' events for home-made simulators, based on atom's speed

    Parameters
    ----------
    u : array, shape (n,m,...,6)
        position/speed vector, according to our vectorization convention
    stop_speed : list
        list of Zones objects targetting speed with actions set to stop

    Returns
    -------
    res: bool
        whether to stop the simulation

    See also
    --------
    atomsmltr.environment.zones
    atomsmltr.simulation.configurator.Configuration.get_stop_zones()
    """
    _, _, _, vx, vy, vz = u.T
    speed = np.array([vx, vy, vz]).T
    res = np.logical_and.reduce([zone.in_zone(speed) for zone in stop_speed])
    res = res
    return res


class RK4(Simulation):
    """A homemade simulator based on fourth order Runge-Kutta method

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider for the simulation

    References
    ----------
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    """

    def __init__(
        self,
        config: Configuration = None,
    ):
        super(RK4, self).__init__(config)

    def get_force(self, u):
        force = get_force_vec(u, self.config)
        return force

    def dudt(self, t, u):
        F = self.get_force(u)
        _, _, _, vx, vy, vz = u.T
        dx, dy, dz = vx, vy, vz
        dvx, dvy, dvz = F.T / self.config.atom.mass
        res = np.array([dx, dy, dz, dvx, dvy, dvz]).T
        return res

    def _integrate(self, u0, t):
        # - u0 to array
        u = np.asanyarray(u0)
        # - get stop events
        events = []
        stop_position, stop_speed = self.config.get_stop_zones()
        if stop_position:
            stop_pos = partial(stop_position_event, stop_position=stop_position)
            events.append(stop_pos)
        if stop_speed:
            stop_sp = partial(stop_speed_event, stop_speed=stop_speed)
            events.append(stop_sp)
        # - time
        # TODO : add checks on time
        t = np.asanyarray(t)
        t = np.sort(t)
        dt = np.diff(t)
        # - initialize
        y = np.empty((*u.shape, len(t)))
        y[..., 0] = u
        stop = False
        # - integrate
        i = 1
        u_none = np.full((6,), np.nan)
        for i, (tt, h) in enumerate(zip(t[1:], dt)):
            # check events
            if events:
                for ev in events:
                    test = ev(u)
                    u[np.logical_not(test), :] = u_none
                    if not np.any(test):
                        stop = True
            if stop:
                break

            # perform step
            k1 = self.dudt(tt, u)
            k2 = self.dudt(tt + 0.5 * h, u + 0.5 * k1 * h)
            k3 = self.dudt(tt + 0.5 * h, u + 0.5 * k2 * h)
            k4 = self.dudt(tt + h, u + k3 * h)
            u_new = u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            u = u_new
            y[..., i + 1] = u_new

        if stop:
            y = y[..., : i + 1]
            t = t[: i + 1]

        res = SimRes(t=t, y=y)

        return res

    def _u0_list_checker(self, value):
        if not hasattr(value, "__iter__"):
            raise ValueError("'u0_list' should be an iterable object")
        if value:
            for u0 in value:
                if np.asanyarray(u0).shape != (6,):
                    raise ValueError("'u0_list' should be a list of arrays of size 6")
        return value
