"""
Examples : Chen 2021
=======================

This example provides the configuration for 1D MOT described
in 'insert exact reference here'.


"""

# % IMPORTS


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# % LOCAL IMPORTS

from atomsmltr.environment import GaussianLaserBeam
from atomsmltr.atoms import Strontium
from atomsmltr.simulation import Configuration, ScipyIVP_3D
from atomsmltr.environment.lasers import CircularLeft
from atomsmltr.environment.fields.magnetic import MagneticQuadrupoleZ



# -- Data file extraction

def extract_results(file_path, step_duration=1e-6):
    out = {}
    with file_path.open() as file:
        current_step_time = 0
        for line in file:
            # Nouveau step ?
            if res := re.search(r"step-(\d+), (\d+)", line):
                step = int(res.group(1))
                current_step_time = step * step_duration
            else:
                pattern = r"(\d+),(\d+)\: \(([0-9.e-]*),([0-9.e-]*),([0-9.e-]*)\)"
                if res := re.search(pattern, line):
                    atom_number = int(res.group(2))
                    x, y, z = map(float, res.groups()[2:])
                    point = [step, current_step_time, x, y, z]
                    out.setdefault(atom_number, []).append(point)
    for atom_number, data in out.items():
        out[atom_number] = np.array(data)
    return out


# --- Paths to the .txt files
base_folder = Path(__file__).parents[2]   # <- remonte à "2021-2022"
data_folder = base_folder / "res"
pos_file = data_folder / "1d_mot_pos.txt"
speed_file = data_folder / "1d_mot_vel.txt"

# Data files reading 
pos = extract_results(pos_file, step_duration=1e-6)
speed = extract_results(speed_file, step_duration=1e-6)


# % GENERATE CONFIGURATION


# -- init config with strontium atom
atom = Strontium()

# -- get Strontium main transition information
main = atom.trans["main"]

# -- setup magnetic field: perfect quadrupole with a strong Z-axis in Chen 2021
# Define magnet properties
origin = np.array((0, 0, 0))
gradient = 0.15  # T/m
mag_field = MagneticQuadrupoleZ(origin=origin, slope=gradient, tag="mag_field_1")

# -- setup lasers
# cf. config from Chen 2021
laser_1 = GaussianLaserBeam()
laser_1.direction = (0, 0, 1)
laser_1.waist_position = (0, 0, 0)
laser_1.waist = (1e-2)*np.sqrt(2)
laser_1.power = 3e-2
laser_1.wavelength = 460.862e-9
laser_1.polarization = CircularLeft()
laser_1.tag = "las1"

laser_2 = laser_1.copy()
laser_2.direction = (0, 0, -1)
laser_2.polarization = CircularLeft()
laser_2.tag = "las2"


# -- add everything to the configuration

config = Configuration()
config.atom = atom

# add objects
config += laser_1, laser_2, mag_field

# setup atomlight
config.add_atomlight_coupling("las1", "main", -2*np.pi*12e6)
config.add_atomlight_coupling("las2", "main", -2*np.pi*12e6)

# Simulation
sim = ScipyIVP_3D(config=config,method="BDF")
t = np.linspace(0, 0.05, 50000)

# initial conditions, identical as the one read in the .txt data files
u0_list = [
    (0, 0, -0.0499099853661708, 0, 0, 10.003050647372623),
    (0, 0, -0.04986498251076984, 0, 0, 15.003647252309184),
    (0, 0, -0.04981997886340934, 0, 0, 20.00440973387974),
    (0, 0, -0.0497749741302099, 0, 0, 25.005399809357463),
    (0, 0, -0.049684959442446455, 0, 0, 35.00847571085285),
    (0, 0, -0.049594931251085383, 0, 0, 45.014391091295806),
    (0, 0, -0.049504869919540194, 0, 0, 55.0272973135054),
    (0, 0, -0.049414709890824214, 0, 0, 65.0611303413218),
    (0, 0, -0.04932416087427111, 0, 0, 75.1782284618543),
    (0, 0, -0.04923175148612011, 0, 0, 85.69587524993723),
    (0, 0, -0.04914393729979519, 0, 0, 95.17949630662842),
    (0, 0, -0.04910343845412679, 0, 0, 99.27069528893709),
    (0, 0, -0.04905908519760101, 0, 0, 104.15474339708909),
    
]



sim_results = [sim.integrate(u0, t) for u0 in u0_list]


# plot of both the curves from the data and the simulation
fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

# --- Simulation (plain curves)
for i, res in enumerate(sim_results, start=1):
    ax.plot(res.y[2], res.y[5], '-', color="navy", linewidth=0.8)

# --- data files (dotted curves)

atoms_to_plot=[3,4,5,6,8,10,12,14,16,18,20,21]

for atom in atoms_to_plot:
    if atom in pos and atom in speed:
        z = pos[atom][:, 4]
        vz = speed[atom][:, 4]
        ax.plot(z, vz, '--', color="navy", lw=0.8)

# plotting details
ax.set_xlabel("z (m)")
ax.set_ylabel("vz (m/s)")
ax.set_xlim(-0.04932416087427111, 0.01)
ax.set_ylim(0, 110)
ax.grid()
ax.legend()
plt.show()