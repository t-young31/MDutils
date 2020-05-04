"""
Calculate the RDF M - O RDF from a trajectory given as xyzs. The metal atom must be the first in the file


first argument is the filename the second is the box size in angstroms
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import crdfgen
from scipy.integrate import trapz

mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize']= 5
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['axes.linewidth'] = 1


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", action='store', type=str, help='Trajectory filename (.xyz,)')
    parser.add_argument('elem_1', action='store', type=str, help='Atomic symbol of element 1')
    parser.add_argument('elem_2', action='store', type=str, help='Atomic symbol of element 2')

    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Generate a RDF plot')

    parser.add_argument('-l', '--boxlength', action='store', type=float, help='Length of the box in Å')
    parser.add_argument('-f', '--firstframe', action='store', type=int, default=0,
                        help='First frame of the trajectory to calculate the RDF from')
    parser.add_argument('-w', '--binwidth', action='store', type=float, help='Size of the bins to plot the RDF in')

    return parser.parse_args()


def get_n_atoms(traj_file_lines):
    """From the trajectory file lines extract the number of atoms, which should be the first item in the first line"""

    try:
        return int(traj_file_lines[0].split()[0])

    except ValueError:
        exit('It looks like the trajectory is malformatted')


def get_elem1_elem2_ids(xyzs, elem1_name, elem2_name):
    """
    From a set of xyzs get the indexes of element 1 and element 2 as two lists

    :param xyzs: (list(list))
    :param elem1_name: (str)
    :param elem2_name: (str)
    :return: (list, list)
    """

    elem1_ids, elem2_ids = [], []

    for i, xyz in enumerate(xyzs):
        if xyz[0].lower() == elem1_name.lower():
            elem1_ids.append(i)

        if xyz[0].lower() == elem2_name.lower():
            elem2_ids.append(i)

    return elem1_ids, elem2_ids


def get_rdf_arrays(xyz_traj_filename, elem1, elem2, box_length, bin_size, first_frame):
    """
    From an MD(/MC) trajectory filename compute the radial distribution function (RDF) from elem1–elem2 e.g. Pd–O.
    Note the whole file will be read into memory, which may be slow/impossible if the trajectory is large

    :param xyz_traj_filename: (str)
    :param elem1: (str)
    :param elem2: (str)
    :param box_length: (float) Å
    :param bin_size: (float) Å
    :param first_frame: (int) first frame of the trajectory to read
    :return:
    """
    try:
        box_length = float(box_length)
        bin_size = float(bin_size)
    except ValueError:
        exit('Box length and bin size MUST be numbers')

    if not os.path.exists(xyz_traj_filename):
        exit(f'Could not open {xyz_traj_filename}. Please make sure it exists')

    traj_file_lines = open(xyz_traj_filename, 'r').readlines()
    n_atoms = get_n_atoms(traj_file_lines)

    # Iterate from the first frame
    n = first_frame

    # Set up the lists of the bin edges and the total frequency of atoms found
    n_bins, bin_edges = int(box_length / (2 * bin_size)), None
    cummulative_hist = np.zeros(n_bins)

    while n*(n_atoms + 2) < len(traj_file_lines):
        xyzs = []
        for line in traj_file_lines[2 + n * (n_atoms + 2):n_atoms + 2 + n * (n_atoms + 2)]:
            atom_label, x, y, z = line.split()[:4]
            xyzs.append([atom_label, float(x), float(y), float(z)])

        # If the first trajectory point, get the element ids of elem1 and elem2 which should not change
        if n == first_frame:
            elem1_ids, elem2_ids = get_elem1_elem2_ids(xyzs, elem1_name=elem1, elem2_name=elem2)

        # Use Cython extension to construct the supercell and calculate the distances between elem1 – elem2
        dists = crdfgen.get_distances(xyzs, elem1_ids, elem2_ids, box_length)

        # Histogram the distances
        hist, bin_edges = np.histogram(dists, bins=n_bins, range=(0.0, box_length / 2.0))

        cummulative_hist += np.array(hist)
        n += 1

    average_hist = cummulative_hist / (n - first_frame)

    # Frequencies and bin edges -> r and densities for plotting p(r) vs r
    r_vals = [(bin_edges[i] + bin_edges[i + 1]) / 2.0 for i in range(len(bin_edges) - 1)]

    # Divide the frequency by the volume between r and r + dr to get the density
    rho_vals = [average_hist[i] / (4.0 / 3.0 * np.pi * (bin_edges[i+1] ** 3 - bin_edges[i] ** 3)) for i in
                range(len(average_hist))]

    # Normalise by the total density to get the pair correlation function
    total_density = sum(average_hist) / (4.0 / 3.0 * np.pi * bin_edges[-1]**3)
    g_vals = np.array(rho_vals) / total_density

    return r_vals, g_vals, total_density


def get_int_r(gs, rho, rs):
    """Get the integral of the pair correlation function, as a function of distance"""
    integrals = []

    integrand = [rs[i]**2 * gs[i] for i in range(len(rs))]

    for i in range(len(rs)):
        integral = rho * 4 * np.pi * trapz(integrand[:i], rs[:i])
        integrals.append(integral)

    return integrals


def main():
    args = get_args()
    rs, gs, rho = get_rdf_arrays(xyz_traj_filename=args.filename, elem1=args.elem_1, elem2=args.elem_2,
                                 box_length=args.boxlength, bin_size=args.binwidth, first_frame=args.firstframe)

    if args.plot:
        fig, ax = plt.subplots()
        ax.plot(rs, gs, lw=1.5)
        ax.set_ylabel('$g(r)$')

        ax2 = ax.twinx()
        ax2.plot(rs, get_int_r(gs, rho, rs), ls='--', c='k')
        ax2.set_ylabel('int($g(r)$)')
        ax2.set_ylim(-0.05, 10.0)

        ax.set_xlabel(f'$r$({args.elem_1}–{args.elem_2}) / Å')
        ax.set_xlim(0.0, args.boxlength/2.0)
        ax.set_ylim(-0.01, 2.0)
        plt.savefig('rdf.png', dpi=300)

    print('r:    ', [np.round(r, 4) for r in rs], sep='\n')
    print()
    print('g(r): ', [np.round(d, 4) for d in gs], sep='\n')

    return None


if __name__ == '__main__':

    main()
