from ase.io import write
from ase.io.trajectory import Trajectory
import argparse
import os


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", action='store', type=str, help='Trajectory filename (.traj,)')
    parser.add_argument('-nw', '--nowrap', action='store_true', default=False, help='Wrap the positions into the box')
    parser.add_argument('-s', '--stride', action='store', type=int, default=1,
                        help='Trajectory will be written in jumps of -s. Default = 1 (every timestep)')

    return parser.parse_args()


def main():

    args = get_args()

    if not os.path.exists(args.filename):
        exit(f'Trajectory file {args.filename} doesn\'t  exist')

    if not args.filename.endswith('.traj'):
        exit('Trajectory must be an ASE .traj file')

    out_filename = args.filename.replace('.traj', '.xyz')

    # Make sure the file is empty
    open(out_filename, 'w').close()

    # Convert the trajectory to a .xyz file
    read_traj = Trajectory(args.filename, 'r')

    for timestep in range(len(read_traj)):

        if timestep % args.stride == 0:

            atoms = read_traj[timestep]
            if args.nowrap is False:
                atoms.wrap()

            write(out_filename, atoms, format='xyz', append=True)

    return None


if __name__ == '__main__':

    main()
