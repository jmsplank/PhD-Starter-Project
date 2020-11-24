"""Matrix rotation to move +z to some arbitrary vector.
Only placed here to avoid circular imports.
"""
import numpy as np


def rot_to_b(B, param):
    """Rotate param so that axis 2 (Z) points
    in direction of B

    IN:
        B:      np.array(3)
                    (x,y,z) coords of vector
        param:  np.array(n,n,n)
                    3D array to rotate.
    """
    # Normalize B
    b = 1.0 / np.linalg.norm(B) * B

    # Find a vector perpendicular to B
    # Assume x=1, y=0, z=z
    v1 = np.array([1, 0, -b[0] / b[2]])
    v1 = 1.0 / np.linalg.norm(v1) * v1

    # Find vector perpendicular to both b and v1
    v2 = np.cross(b, v1)
    v2 = 1.0 / np.linalg.norm(v2) * v2
    # New orthonormal basis b, v1, v2 where v1, v2 arbitrary
    basis = np.column_stack((b, v1, v2))
    # Inverse of new basis
    ibasis = np.linalg.inv(basis)

    # Calculate components in new basis
    return np.matmul(ibasis, param)