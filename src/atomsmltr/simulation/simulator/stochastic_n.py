"""Home-made deterministic integrators
=========================================

Implements homemade integrators for stochastic systems, that is, taking into
acount diffusion due to photon scattering and spontaneous emission.
"""

# % IMPORTS
import numpy as np
import scipy.constants as csts
from functools import partial

# % LOCAL IMPORTS
from .simbase import Simulation, SimRes, get_force_vec
from .deterministic import CustomSimulationBase
from ..configurator import Configuration

Gamma_ytterbium = 2 * np.pi * 28.9e6  # rad/s


# % HOME-MADE SIMULATORS


# --- atoms_pointing_origin_speed : always return a (n,6) array ---
def atoms_pointing_origin_speed(theta, step, min_speed, max_speed):
    """Generates n vectors np.array([x,y,z,vx,vy,vz]) for atoms with initial colatitude angle "theta",
    with initial speed (norm) ranging from "min_speed" to "max_speed" with step "step"
    We suppose the azimuthal angle null as the studied phenomenon is invariant by rotation around axis z

    Parameters
    ----------
    theta : float
        initial colatitude angle (between the initial speed vector and z-axis)
    step : float
        step between each initial speed norm of the generated vectors
    min_speed : float
        initial speed norm of the first generated vector
    max_speed : float
        initial speed norm of the last generated vector

    Returns
    -------
    vectors : array
        shape (n,6) where n is the number of atoms and vectors[i] is the position/speed vector of atom i
    """
    n = round((max_speed - min_speed) / step)
    speed_list = np.zeros(n)
    speed_list[0] = min_speed
    # each vector has an additional "step" norm than the previous one
    for i in range(1, n):
        speed_list[i] = speed_list[i - 1] + step
    vectors = np.zeros((n, 6))
    # vectors[:,1] = vectors[:4] = 0 always, as the azitumal angle is null
    vectors[:, 3] = -speed_list * np.sin(theta)
    vectors[:, 5] = -speed_list * np.cos(theta)
    # atoms are initially placed at 1 cm from the origin
    vectors[:, 0] = np.sin(theta) * 0.01
    vectors[:, 2] = np.cos(theta) * 0.01
    return vectors


# --- random_unit_vector : always return a (n,3) array ---
def random_unit_vector(shape=(1,)):
    """Generates a random unit vector for each requested sample.

    Parameters
    ----------
    shape : tuple or int, optional
        number of vectors to generate (returns (n,3)), by default (1,)

    Returns
    -------
    vec : array
        shape (n,3) where n = np.prod(shape) if shape is tuple/int > 1, or (1,3) if scalar
    """
    rng = np.random.default_rng()
    # allow integer input
    if isinstance(shape, int):
        size = (shape,)
    else:
        size = tuple(shape) if isinstance(shape, tuple) else (shape,)
    # produce 1-D length n arrays
    n = int(np.prod(size))
    phi = rng.uniform(low=0, high=2 * np.pi, size=n)
    costheta = rng.uniform(low=-1, high=1, size=n)
    sintheta = np.sqrt(np.maximum(0.0, 1 - costheta**2))
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = costheta
    vec = np.stack([x, y, z], axis=1)  # (n,3)
    return vec


class RK4_Stochastic_n(CustomSimulationBase):
    """A homemade simulator based on fourth order Runge-Kutta method

    Parameters
    ----------
    config : Configuration, optional
        the configuration to consider for the simulation

    References
    ----------
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    """

    def __init__(self, config: Configuration = None):
        super(RK4_Stochastic_n, self).__init__(config)

    def dudt(self, t, u):
        """Computes the deterministic part of the movement due to photon scattering

        Parameters
        ----------
        t : float
            the instant at which the deterministic part of the motion is calculated
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the
            position/speed vector of atom i at time t

        Returns
        -------
        res : array
            shape (n,6) where n is the number of atoms and res[i] is the
            speed/acceleration vector of atom i at time t
        """
        F = self.get_force(u)
        _, _, _, vx, vy, vz = u.T
        dx, dy, dz = vx, vy, vz
        dvx, dvy, dvz = F.T / self.config.atom.mass
        res = np.array([dx, dy, dz, dvx, dvy, dvz]).T
        return res

    def du_fluct(self, t, u, dt):
        """Computes the stochastic part of the movement due to spontaneous emission and random absorption

        Parameters
        ----------
        t : float
            the instant at which the deterministic part of the motion is calculated
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the
            position/speed vector of atom i at time t
        dt : float
            time step between 2 successive instants

        Returns
        -------
        res : array
            shape (n,6) where n is the number of atoms and res[i] is the
            speed/acceleration vector of atom i at time t
        """
        n_atoms = u.shape[0]
        positions = u[:, :3]
        velocities = u[:, 3:]
        dv_tot = np.zeros((n_atoms, 3))
        # get scattering info once for the whole batch
        _, scatt_list = get_force_vec(u, self.config, return_list=True)
        Gamma = Gamma_ytterbium
        # compute the scattering rate Ri for each channel (as array shape (n_atoms,))
        Ri_arrays = []
        k_list = []
        dir_list = []
        for sc in scatt_list:
            Ri = np.asarray(sc["rate"])
            # if scalar -> broadcast to shape (n_atoms,)
            if Ri.ndim == 0:
                Ri = np.full(n_atoms, float(Ri))
            Ri_arrays.append(Ri)
            k_list.append(np.asarray(sc["k"]))
            direction = np.asarray(sc["unit_vector"])
            # ensure direction has shape (n_atoms,3)
            if direction.ndim == 1 and direction.size == 3:
                direction = np.broadcast_to(direction, (n_atoms, 3))
            elif (
                direction.ndim == 2
                and direction.shape[0] == n_atoms
                and direction.shape[1] == 3
            ):
                pass
            else:
                # try to reshape or broadcast; fallback to zeros
                try:
                    direction = np.reshape(direction, (n_atoms, 3))
                except Exception:
                    direction = np.zeros((n_atoms, 3))
            dir_list.append(direction)
        # sum scattering rates per atom across channels -> shape (n_atoms,)
        Ri_sum = np.zeros(n_atoms)
        for Ri in Ri_arrays:
            Ri_sum = Ri_sum + Ri
        # avoid division by zero for small Ri_sum (if the atom is initially far from the MOT)
        safe_Ri_sum = np.where(Ri_sum <= 0, 0.0, Ri_sum)
        # excited state population per atom
        rho_ee = np.where(
            safe_Ri_sum == 0, 0.0, safe_Ri_sum / (Gamma + 2.0 * safe_Ri_sum)
        )
        N_gamma = dt * rho_ee * Gamma
        # creates a randomdirection vector
        rng = np.random.default_rng()
        # loop over channels (still a small loop over beams, not atoms)
        for idx, Ri in enumerate(Ri_arrays):
            k_val = k_list[idx]
            # shape (n_atoms,3)
            direction = dir_list[idx]
            # Ni per atom for this channel
            if np.all(Ri_sum == 0):
                Ni = np.zeros_like(Ri_sum)
            else:
                Ni = np.where(Ri_sum == 0, 0.0, (Ri / Ri_sum) * N_gamma)
            # draw Poisson samples per atom
            N_i_tilde = rng.poisson(Ni)
            # absorption force: per-atom per-channel
            # (n_atoms,1) * scalar(ki*hbar) * (n_atoms,3) -> (n_atoms,3)
            F_abs = (N_i_tilde[:, None] * (k_val * csts.hbar)) * direction
            dv_tot += F_abs / self.config.atom.mass
            # spontaneous emission: random directions per atom
            rd_vec = random_unit_vector(shape=(n_atoms,))  # (n_atoms,3)
            gauss_moyenne = np.sqrt(np.maximum(N_i_tilde, 0) / 3.0) * (
                csts.hbar * k_val
            )
            F_spontem = gauss_moyenne[:, None] * rd_vec
            dv_tot += F_spontem / self.config.atom.mass
        # build output (n_atoms,6)
        dx = np.zeros(n_atoms)
        dy = np.zeros(n_atoms)
        dz = np.zeros(n_atoms)
        dvx, dvy, dvz = dv_tot.T
        res = np.stack([dx, dy, dz, dvx, dvy, dvz], axis=1)
        return res

    def _iterate(self, t, u, dt):
        """Computes the stochastic and the deterministic part of the movement of the atom

        Parameters
        ----------
        t : float
            the instant at which the forces applied to the atom are calculated
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the
            position/speed vector of atom i at time t
        dt : float
            time step between 2 successive instants

        Returns
        -------
        du_tot : array
            shape (n,6)
        """
        # deterministic part
        k1 = self.dudt(t, u)
        k2 = self.dudt(t + 0.5 * dt, u + 0.5 * k1 * dt)
        k3 = self.dudt(t + 0.5 * dt, u + 0.5 * k2 * dt)
        k4 = self.dudt(t + dt, u + k3 * dt)
        du_det = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # stochastic part
        du_fluct = self.du_fluct(t, u, dt)
        # sum of the two contributions
        du_tot = du_det + du_fluct
        return du_tot

    def resolve_stochastic(self, t, u):
        """Effectively solves the movement and speed equation over a period (0, t_final) by iterating
        the previous method

        Parameters
        ----------
        t : array
            shape (m) where m is the total amount of instant used to calculate the movement,
            typically t[0] = 0 and t[m] = t_final
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the initial
            position/speed vector of atom i.

        Returns
        -------
        trajectories : array
            shape (len(t),n,6) array listing the position/speed vectors of all n atoms, at every time t
            ranging from t_start to t_final
        """
        n_atoms = u.shape[0]
        trajectories = np.zeros((len(t), n_atoms, 6))
        trajectories[0] = u
        for i in range(1, len(t)):
            du = self._iterate(t[i], trajectories[i - 1], t[i] - t[i - 1])
            trajectories[i] = trajectories[i - 1] + du
        return trajectories

    def instant_temperature_per_atom(self, t, u):
        """Computes the instant temperature of each atom in the system at each time t

        Parameters
        ----------
        t : array
            shape (m) where m is the total amount of instant used to calculate the movement,
            typically t[0] = 0 and t[m] = t_final
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the initial
            position/speed vector of atom i.

        Returns
        -------
        instant_temperature : array
            shape (len(t),n) array listing the instant temperature of all n atoms, at every time t
            ranging from t_start to t_final
        """
        trajectories = self.resolve_stochastic(t, u)
        start_frame = int(5 * len(t) / 6)
        n_record_frames = max(1, len(t) - start_frame)
        instant_temperature = np.zeros((n_record_frames, u.shape[0]))
        for i in range(start_frame, len(t)):
            instant_speed = trajectories[i, :, 3:6]
            quadratic_speed = np.sum(instant_speed**2, axis=1)
            instant_temperature[i - start_frame] = (
                self.config.atom.mass * quadratic_speed
            ) / (3.0 * csts.Boltzmann)
        return instant_temperature

    def doppler_temperature_temporal_mean(self, t, u):
        """Computes the mean temperature of the atoms in the system at a time t

        Parameters
        ----------
        t : array
            shape (m) where m is the total amount of instant used to calculate the movement,
            typically t[0] = 0 and t[m] = t_final
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the  initial
            position/speed vector of atom i.

        Returns
        -------
        mean_temperature : array
            shape (len(t),) array listing the mean temperature of the system at every time t,
            ranging from t_start to t_final
        """
        instant_temperature = self.instant_temperature_per_atom(t, u)
        mean_temperature = np.mean(instant_temperature, axis=1)
        return mean_temperature

    def speed_modulus(self, t, u):
        """Computes the modulus of the instant velocity of each atom at every time t,
        ranging from t_start to t_final.

        Parameters
        ----------
        t : array
            shape (m) where m is the total amount of instant used to calculate the movement,
            typically t[0] = 0 and t[m] = t_final
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the  initial
            position/speed vector of atom i.

        Returns
        -------
        speed_modulus : array
            shape (len(t),n) array listing the modulus of the instant velocity for every atom
            in the system at each instant t, ranging from t_start to t_final
        """
        trajectories, _, _ = self.resolve_stochastic(t, u)
        vx = trajectories[:, :, 3]
        vy = trajectories[:, :, 4]
        vz = trajectories[:, :, 5]
        speed_modulus = np.sqrt(vx**2 + vy**2 + vz**2)
        return speed_modulus

    def position_modulus(self, t, u):
        """Computes the modulus of the instant position of each atom at every time t,
        ranging from t_start to t_final.

        Parameters
        ----------
        t : array
            shape (m) where m is the total amount of instant used to calculate the movement,
            typically t[0] = 0 and t[m] = t_final
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the  initial
            position/speed vector of atom i.

        Returns
        -------
        position_modulus : array
            shape (len(t),n) array listing the modulus of the instant position for every atom
            in the system at each instant t, ranging from t_start to t_final
        """
        trajectories, _, _ = self.resolve_stochastic(t, u)
        x = trajectories[:, :, 0]
        y = trajectories[:, :, 1]
        z = trajectories[:, :, 2]
        position_modulus = np.sqrt(x**2 + y**2 + z**2)
        return position_modulus

    def min_non_catching_speed(self, t, u, threshold=1.0):
        """Calculates the minimum speed necessary for the atom not to be caught by the MOT.

        Parameters
        ----------
        t : array
            shape (m) where m is the total amount of instant used to calculate the movement,
            typically t[0] = 0 and t[m] = t_final
        u : array
            shape (n,6) where n is the number of atoms and u[i] is the  initial
            position/speed vector of atom i.
        threshold : float
            arbitrary float number, if the average asymptotic speed of the atom is abive, the
            atom is not considerer caught

        Returns
        -------
        min(non_captured), non_captured : float, array
            non_captured is of shape (,) and list the initial speed modulus of the atoms
            that have not been caught by the MOT.
        """
        t_min = round((5 * len(t)) / 6)
        quadratic_velocities = self.speed_modulus(t, u)
        non_captured = []
        for i in range(len(u)):
            instant_velocity = quadratic_velocities[t_min:, i]
            mean_velocity = np.mean(instant_velocity)
            if mean_velocity > threshold:
                initial_velocity = np.linalg.norm(u[i, 3:6])
                non_captured.append(initial_velocity)
        if non_captured:
            return min(non_captured), non_captured
        else:
            print("All the atoms in this speed range have been caught.")
            return None

    def theta_min_speed(
        self, t, theta, step, min_speed, max_speed, number_of_iterations
    ):
        """Generates the initial vectors using the atoms_pointing_origins_speed() method,
        then calculates the average (over the number_of_iterations) of the minimum speed
        required for the atom not to be caught given theta.

        Parameters
        ----------
        t : array
            shape (m) where m is the total amount of instant used to calculate the movement,
            typically t[0] = 0 and t[m] = t_final
        theta : float
            initial colatitude angle (between the initial speed vector and z-axis)
        step : float
            step between each initial speed norm of the generated vectors
        min_speed : float
            initial speed norm of the first generated vector
        max_speed : float
            initial speed norm of the last generated vector
        number_of_iterations : int
            number of loops carried out by the method --> higher number_of_iterations improve precision

        Returns
        -------
        mean_speed_theta : float
            average minimum speed required for the atom not to be caught given theta


        WARNING :
            high number_of_iterations may greatly increase the time necessary to run the code
        """
        u_initials = atoms_pointing_origin_speed(theta, step, max_speed, min_speed)
        mean_speed_theta = 0
        for i in range(number_of_iterations):
            min_speed_theta, _ = self.min_catching_speed(t, u_initials)
            mean_speed_theta += min_speed_theta
        mean_speed_theta = mean_speed_theta / number_of_iterations
        return mean_speed_theta
