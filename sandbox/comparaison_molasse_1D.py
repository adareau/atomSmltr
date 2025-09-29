# % IMPORTS

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# % LOCAL IMPORTS

from atomsmltr.environment import GaussianLaserBeam
from atomsmltr.atoms import Rubidium
from atomsmltr.simulation import Configuration, ScipyIVP_3D
from atomsmltr.environment.lasers import CircularLeft, CircularRight

# % DATA EXTRACTION FUNCTION
def extract_results(file_path, step_duration=1e-6):
    out = {}
    with file_path.open() as file:
        current_step_time = 0
        for line in file:
            # is it a new step ?
            if res := re.search(r"step-(\d+), (\d+)", line):
                step = int(res.group(1))
                current_step_time = step * step_duration
            else:
                pattern = r"(\d+),(\d+)\: \(([0-9.e-]*),([0-9.e-]*),([0-9.e-]*)\)"
                if res := re.search(pattern, line):
                    atom_number = int(res.group(2))
                    vx = float(res.group(3))
                    vy = float(res.group(4))
                    vz = float(res.group(5))
                    point = [step, current_step_time, vx, vy, vz]
                    if atom_number in out:
                        out[atom_number].append(point)
                    else:
                        out[atom_number] = [point]
    for atom_number, data in out.items():
        out[atom_number] = np.array(data)
    return out


# % PATH TO THE .TXT DATA FILES
script_folder = Path(__file__).parents[2]
data_folder =  script_folder / "res_molasse"
speed_file = data_folder / "1d_mol_vel.txt"


# % RETRIEVE DATA
step_duration = 1e-6  # seconds
speed = extract_results(speed_file, step_duration)


# % GENERATE CONFIGURATION

# -- init config with rubidium atom
atom = Rubidium()


# -- get Rubidium main transition information
main = atom.trans["main"] 


# -- setup lasers
# cf. config from Chen 2021
laser_1 = GaussianLaserBeam()
laser_1.direction = (0, 0, 1)
laser_1.waist_position = (0, 0, 0)
laser_1.waist =  (10e-2)*np.sqrt(2)   # 1/e^2 radius
laser_1.power = 10e-2
laser_1.wavelength=780.241e-9 
laser_1.polarization = CircularLeft()  # circular left
laser_1.tag = "las1"

laser_2 = laser_1.copy() # create a copy
laser_2.direction = (0, 0, -1)  # propagating in opposite direction
laser_2.polarization = CircularLeft()  # circular right
laser_2.tag = "las2"

# -- add everything to the configuration
config = Configuration()
config.atom = atom

# add objects
config += laser_1, laser_2

# setup atomlight
config.add_atomlight_coupling("las1", "main", -2*np.pi*6e6)
config.add_atomlight_coupling("las2", "main", -2*np.pi*6e6)

# -- initial conditions
sim = ScipyIVP_3D(config=config, method ="RK45")
t = np.linspace(0, 0.001, 20000)  # timesteps for integration

u0_list=[
    (0, 0, -0.03, 0, 0, 10),
    (0, 0, -0.03, 0, 0, 11),
    (0, 0, -0.03, 0, 0, 12),
    (0, 0, -0.03, 0, 0, 13),
    (0, 0, -0.03, 0, 0, 14),
    (0, 0, -0.03, 0, 0, 15),
    (0, 0, -0.03, 0, 0, 16),
    (0, 0, -0.03, 0, 0, 17),
    (0, 0, -0.03, 0, 0, 18),
    (0, 0, -0.03, 0, 0, 19),
    (0, 0, -0.03, 0, 0, 20),
    (0, 0, -0.03, 0, 0, 21),
    (0, 0, -0.03, 0, 0, 22),
    (0, 0, -0.03, 0, 0, 23),
    (0, 0, -0.03, 0, 0, 24),
    (0, 0, -0.03, 0, 0, 25),
    (0, 0, -0.03, 0, 0, 26),
    (0, 0, -0.03, 0, 0, 27),
    (0, 0, -0.03, 0, 0, 28),
    (0, 0, -0.03, 0, 0, 29),

]

sim_results = [sim.integrate(u0, t) for u0 in u0_list]


# % PLOT

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

# Plain curves - atomSmltr

for i, res in enumerate(sim_results, start=1):
    ax.plot(t*1e6, res.y[5], '-', color="navy", linewidth=0.8)

# DDotted curves - AtomECS


for atom, data in speed.items() : 
    t=data[:,1] * 1e6
    vz=data[:,4]
    plt.plot(t,vz,'--',color="navy",lw=0.8)


plt.xlabel("t (μs)")
plt.ylabel("vz (m/s)")
plt.xlim(0, 1000)
plt.ylim(0, 30)
plt.grid()
plt.show()

