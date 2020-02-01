"""
Calculate a diffusion histogram for waters molecules from an ASE trajectory. Diffusion coefficients are calculated
using the velocity autocorrelation function


"""
from ase.io.trajectory import Trajectory
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", action='store', type=str, help='Trajectory filename (.traj,)')
    parser.add_argument("-pv", '--plotVACF', action='store_true', default=False, help='Plot the VACF')

    return parser.parse_args()


def get_waters_idxs(atoms, max_oh_bond_length=1.2):
    """
    Use networkX to find the connected water molecules. The connected component of the full graph must have
    3 atoms where the O–H bonds are separated by at most max_oh_bond_length Å

    :param atoms: (ASE.Atoms)
    :param max_oh_bond_length: (float) Å
    :return: (list(list))
    """

    atom_symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    n_atoms = len(atom_symbols)

    full_graph = nx.Graph()
    # Add the oxygen atoms first so the water indexes have the form O, H, H
    [full_graph.add_node(i, atom_label=atom_symbols[i]) for i in range(n_atoms) if atom_symbols[i] == 'O']
    [full_graph.add_node(i, atom_label=atom_symbols[i]) for i in range(n_atoms) if atom_symbols[i] != 'O']

    dist_mat = distance_matrix(positions, positions)

    # Iterate over all unique pairs and add edges between O-H atoms
    for i, atom in enumerate(positions):
        for j, alt_atom in enumerate(positions):
            if i > j:
                if ((atom_symbols[i] == 'O' and atom_symbols[j] == 'H') or
                   (atom_symbols[i] == 'H' and atom_symbols[j] == 'O')):

                    if dist_mat[i, j] <= max_oh_bond_length:
                        full_graph.add_edge(i, j)

    connected_waters = [list(sorted(mol)) for mol in nx.connected_components(full_graph) if len(mol) == 3]

    print(f'Found {len(connected_waters)} connected water molecules')
    return connected_waters


def get_water_com_traj_velocities(trajectory):
    """
    From an ASE trajectory extract the velocities with shape (steps x n_atoms x 3) and the indexes of the
    water atoms

    :param trajectory: (ASE.Trajectory)
    :return: (np.ndarray, list(list))
    """

    traj_vels = []
    waters_idxs = []

    for timestep in range(len(trajectory)):

        if timestep == 0:
            waters_idxs = get_waters_idxs(atoms=trajectory[timestep])

        velocities = trajectory[timestep].get_velocities()
        com_velocities = get_water_com_velocities(vels=velocities, waters_idxs=waters_idxs)

        traj_vels.append(com_velocities)

    print(f'Generated COM velocities for {len(waters_idxs)} waters for {len(traj_vels)} timesteps')
    return np.array(traj_vels)


def get_water_com_velocities(vels, waters_idxs, o_mass=16.0, h_mass=1.01):
    """
    Calculate the center of mass velocity for a water molecules

    :param vels: (np.ndarray) n_atoms x 3                       x,y,z
    :param waters_idxs: (list(list)) n_waters x 3               O,H,H
    :param o_mass: (float) amu
    :param h_mass: (float) amu
    :return: (np.ndarray) n_waters x 3                          x,y,z
    """

    com_velocities = []
    h2o_mass = o_mass + 2 * h_mass

    for water_idxs in waters_idxs:

        o_idx, h1_idx, h2_idx = water_idxs

        # Calculate the center of mass velocity as Σ_i m_i v_i / M   where M = Σ_i m_i
        com_vel = (o_mass * vels[o_idx] + h_mass * (vels[h1_idx] + vels[h2_idx])) / h2o_mass
        com_velocities.append(com_vel)

    return np.array(com_velocities)


def calc_vacf(traj_vels, mol_id=None, plot=False):
    """
    Calculate the velocity auto-correlation function (VACF)

    phi(t) = <v(t).v(t0)>  where the ensemble average is taken over the possible time origins

    :param traj_vels:
    :param mol_id:
    :param plot:
    :return:
    """

    n_steps = traj_vels.shape[0]
    n_mols = traj_vels.shape[1]
    vacf = np.zeros(n_steps - 1)

    for time_origin in np.arange(n_steps - 1):
        for t in range(time_origin + 1, n_steps):

            # Compute cumulative VACF value for time = t either for a single molecule(/index) or averaged
            # over all velocities

            if mol_id is not None:
                vacf[t - time_origin - 1] += np.dot(traj_vels[t, mol_id], traj_vels[time_origin, mol_id])

            else:
                vdotvs = [np.dot(traj_vels[t, i], traj_vels[time_origin, i]) for i in range(n_mols)]
                vacf[t - time_origin - 1] += np.average(np.array(vdotvs))

    vacf /= (n_steps - 1)       # Divide the cumulative VACF by the number of time origins taken

    if plot:
        plt.scatter(np.arange(n_steps - 1), vacf, s=2)
        plt.ylabel('VACF')
        plt.xlabel('Timestep')
        plt.savefig('vacf.png')

    return vacf


def calc_diffusion_coefficient(vacf_):
    return (1.0 / 3.0) * np.trapz(vacf_, dx=1.0)


if __name__ == '__main__':

    args = get_args()

    traj_velocities = get_water_com_traj_velocities(trajectory=Trajectory(args.filename, 'r'))
    vacf = calc_vacf(traj_vels=traj_velocities, plot=args.plotVACF)
    d = calc_diffusion_coefficient(vacf_=vacf)

    print(f'Diffusion coefficent = {d} au.')
