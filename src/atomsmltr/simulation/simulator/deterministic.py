"""Home-made deterministic integrators
=========================================

Implements homemade integrators for deterministic systems, that is, not taking into
acount diffusion due to photon scattering.
"""

# % IMPORTS
import numpy as np
from functools import partial
from abc import abstractmethod

# % LOCAL IMPORTS
from .simbase import Simulation, SimRes, get_force_vec
from ..configurator import Configuration


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
    res = np.logical_and.reduce([zone.get_value(position) for zone in stop_position])
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
    res = np.logical_and.reduce([zone.get_value(speed) for zone in stop_speed])
    res = res
    return res


class CustomSimulationBase(Simulation):
    """A Base for home-made deterministic simulations

    Not meant to be used directly, just gathering common method
    """

    def __init__(
        self,
        config: Configuration = None,
    ):
        super(CustomSimulationBase, self).__init__(config)

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

    @abstractmethod
    def _iterate(self, t, u, dt):
        """returns the evolution du of u between t and t+dt"""

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
            u = u + self._iterate(tt, u, h)
            y[..., i + 1] = u

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


class RK4(CustomSimulationBase):
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

    def _iterate(self, t, u, dt):
        """returns the evolution du of u between t and t+dt
        Here we use the fourth order Runge-Kutta method
        """
        # perform step
        k1 = self.dudt(t, u)
        k2 = self.dudt(t + 0.5 * dt, u + 0.5 * k1 * dt)
        k3 = self.dudt(t + 0.5 * dt, u + 0.5 * k2 * dt)
        k4 = self.dudt(t + dt, u + k3 * dt)
        du = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return du


class Euler(CustomSimulationBase):
    """A homemade simulator based on Euler's method

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider for the simulation

    References
    ----------
    TODO: put here

    """

    def __init__(
        self,
        config: Configuration = None,
    ):
        super(Euler, self).__init__(config)

    def _iterate(self, t, u, dt):
        """returns the evolution du of u between t and t+dt
        Here we use the Euler method
        """
        F = self.get_force(u)
        _, _, _, vx, vy, vz = u.T
        dvx, dvy, dvz = F.T / self.config.atom.mass * dt
        dx = vx * dt + 0.5 * dvx * dt
        dy = vy * dt + 0.5 * dvy * dt
        dz = vz * dt + 0.5 * dvz * dt
        du = np.array([dx, dy, dz, dvx, dvy, dvz]).T
        return du


class VelocityVerlet(CustomSimulationBase):
    """A homemade simulator based on velocity verlet method

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider for the simulation

    References
    ----------
    TODO: put here

    """

    def __init__(
        self,
        config: Configuration = None,
    ):
        super(VelocityVerlet, self).__init__(config)

    def _iterate(self, t, u, dt):
        """returns the evolution du of u between t and t+dt
        Here we use the Velocity verlet method
        """
        # 1) compute next position
        F = self.get_force(u)
        x, y, z, vx, vy, vz = u.T
        ax, ay, az = F.T / self.config.atom.mass
        dx = vx * dt + 0.5 * ax * dt**2
        dy = vy * dt + 0.5 * ay * dt**2
        dz = vz * dt + 0.5 * az * dt**2
        # 2) compute next speed
        u_next = np.array([x + dx, y + dy, z + dz, vx, vy, vz]).T
        F_next = self.get_force(u_next)
        ax_next, ay_next, az_next = F_next.T / self.config.atom.mass
        dvx = 0.5 * (ax + ax_next) * dt
        dvy = 0.5 * (ay + ay_next) * dt
        dvz = 0.5 * (az + az_next) * dt
        du = np.array([dx, dy, dz, dvx, dvy, dvz]).T
        return du
