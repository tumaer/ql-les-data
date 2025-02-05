"""Convert files from a directory of .h5 files to .vtk files."""

import argparse

from jax_sph.io_state import write_vtks_from_h5s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .h5 files to .vtk files.")
    parser.add_argument("--path", type=str, 
                        help="Path to the directory with the .h5 files.")
    args = parser.parse_args()

    write_vtks_from_h5s(args.path)