# -*- coding: utf-8 -*-
"""Defines the main simulator classe
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


def get_force_vec(pos_speed_vector, config):
    """Computes the force on an atom, by adding all radiations pressures
    Note : we could refine the modul using a rate model, as in atomECS

    The function is vectorized to be compatible with Scipy's `solve_ivp`

    The position/speed vector 'pos_speed_vector' should be of shape (6,) or (6,k)
    with the first dimension containing (x, y, z, vx, vy, vz) in the lab frame

    Args:
        pos_speed_vector (array like): position and speed, shape (6,) or (6,k)
        config (Configuration): a Configuration object for the simulation

    Returns:
        force (array like): the force, shape (3,) or (3,k)
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

    return force


def get_force_n6_convention(pos_speed_vector, config):
    """Computes the force on an atom, by adding all radiations pressures
    Note : we could refine the modul using a rate model, as in atomECS

    The function is vectorized to be compatible with Scipy's `solve_ivp`

    The position/speed vector 'pos_speed_vector' should be of shape (..,6)
    with the first dimension containing (x, y, z, vx, vy, vz) in the lab frame

    Args:
        pos_speed_vector (array like): position and speed, shape (..,6)
        config (Configuration): a Configuration object for the simulation

    Returns:
        force (array like): the force, shape (...,3)
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


class BaseSimulator(ABC):
    """docstring for Simulator."""

    def __init__(self, config=None):
        super(BaseSimulator, self).__init__()
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


class ScipyIVP_3D(BaseSimulator):
    """docstring for ScipyIVP_3D."""

    def __init__(self, config=None, method="Radau", **solve_ivp_args):
        super(ScipyIVP_3D, self).__init__(config)
        self.solve_ivp_args = solve_ivp_args
        self.method = method

    # -- REQUESTED FUNCTIONS
    def get_force(self, u):
        force = get_force_vec(u, self.config)
        return force

    def get_force_n6(self, u):
        force = get_force_n6_convention(u, self.config)
        return force

    def dudt(self, t, u):
        F = self.get_force(u)
        _, _, _, vx, vy, vz = u
        dx, dy, dz = vx, vy, vz
        dvx, dvy, dvz = F.T / self.config.atom.mass
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
