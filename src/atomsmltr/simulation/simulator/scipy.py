"""Scipy-based integrators
============================

Implements integrator based on scipy's ``solve_ivp`` method.
"""

# % IMPORTS
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

# % LOCAL IMPORTS
from .simbase import Simulation, get_force_vec, _get_force_vec
from ..configurator import Configuration


# % SIMULATOR BASED ON SCIPY'S SOLVE_IVP


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
    res = np.logical_and.reduce([zone.get_value(position) for zone in stop_position])
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
    res = np.logical_and.reduce([zone.get_value(speed) for zone in stop_speed])
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
