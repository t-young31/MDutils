# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.math cimport sqrt, fmin
import numpy as np


cdef double[:] get_dists_to_i(double [:] elem1_coord, double[:, :] elem2_coords,
                              double[:] dists_to_i, double box_length, int n_elem2s):

    cdef int n = 0
    cdef double d, tmp = 0
    cdef int i, j, k, m, o
    cdef double vec[3]

    cdef double half_bl_sq = (box_length / 2.0) * (box_length / 2.0)
    cdef double sq_bl = box_length * box_length

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):

                for m in range(n_elem2s):
                    # Calculate the shift vector
                    vec[0] = (elem2_coords[m, 0] + <double>i * box_length - elem1_coord[0])
                    vec[1] = (elem2_coords[m, 1] + <double>j * box_length - elem1_coord[1])
                    vec[2] = (elem2_coords[m, 2] + <double>k * box_length - elem1_coord[2])

                    # Compute the distance
                    d = 0.0
                    for o in range(3):
                        d += vec[o] * vec[o]

                    # If the elem1--m distance is to the closest version of itself
                    if d < half_bl_sq:
                        dists_to_i[n] = sqrt(d)

                    n += 1

    return dists_to_i

def get_distances(py_xyzs, py_elem1_ids, py_elem2_ids, py_box_length):
    """
    Generate
    """

    cdef double[:, :] elem2_coords = np.array([np.array(line[1:4]) for i, line in enumerate(py_xyzs) if i in py_elem2_ids], dtype='f8')
    cdef double box_length = py_box_length

    all_dists = []
    cdef double[:] elem1_coord
    cdef double[:] dists_to_i
    cdef int n_elem2s = len(py_elem2_ids)

    for py_elem1_id in py_elem1_ids:

        elem1_coord = np.array(py_xyzs[py_elem1_id][1:4], dtype='f8')
        dists_to_i = np.zeros(3**3*len(py_elem2_ids), dtype='f8')
        py_dists = np.asarray(get_dists_to_i(elem1_coord, elem2_coords, dists_to_i, box_length, n_elem2s))

        # Strip all the zeros from the distances
        all_dists += py_dists[py_dists != 0].tolist()

    return all_dists
