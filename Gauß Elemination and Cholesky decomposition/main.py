import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if len(A[0]) != len(b):
        raise ValueError
    # TODO: Perform gaussian elimination
    n = b.size
    for i in range(0, n - 1):
        if use_pivoting:
            for k in range(i + 1, n):
                if np.abs(A[k, i]) > np.abs(A[i, i]):
                    A[[i, k]] = A[[k, i]]
                    b[[i, k]] = b[[k, i]]
                    break

        for j in range(i + 1, n):
            if A[i, i] == 0:
                raise ValueError
            m = A[j, i] / A[i, i]
            A[j, :] = A[j, :] - m * A[i, :]
            b[j] = b[j] - m * b[i]

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    n = b.size
    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    if len(A[0]) != len(b):
        raise ValueError

    # TODO: Initialize solution vector with proper size
    x = np.zeros(n)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    if A[n - 1, n - 1] == 0:
        raise ValueError

    x[n - 1] = b[n - 1] / A[n - 1, n - 1]  # Solve for last entry first
    for i in range(n - 2, -1, -1):  # Loop from the end to the beginning
        sum_ = 0
        for j in range(i + 1, n):  # For known x values, sum and move to rhs
            sum_ = sum_ + A[i, j] * x[j]
        x[i] = (b[i] - sum_) / A[i, i]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape

    if n != m:
        raise ValueError

    t = np.allclose(M, M.T)
    if not t:
        raise ValueError

    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))

    for i in range(n):
        for k in range(0, i + 1):
            if i == k:
                L[i, i] = M[i, i] - sum(L[i][j] * L[i][j] for j in range(i))
                if L[i, i] <= 0:
                    raise ValueError
                L[i][i] = np.sqrt(L[i, i])
            else:
                L[i][k] = (M[i, k] - sum(L[i][j] * L[k][j] for j in range(k)))/L[k, k]

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    (k, ) = b.shape

    if m != n:
        raise ValueError
    if m != k:
        raise ValueError
    if not np.allclose(L, np.tril(L)):
        raise ValueError

    # TODO Solve the system by forward- and backsubstitution

    # for i in range(n):
    # sum = 0
    # for j in range(i):
    #     sum += L[i, j] * x[j]
    # x[i] = (b[i] - sum)/L[i, i]

    frw = (back_substitution(L[::-1, ::-1], b[::-1]))[::-1]

    x = back_substitution(L.T, frw)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_rays*n_shots, n_grid*n_grid))
    # TODO: Initialize intensity vector
    g = np.zeros(n_rays*n_shots)

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.
    intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)

    for x in range(n_shots):
        theta = x * (np.pi / n_shots)
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        for i in range(len(ray_indices)):
            L[x*n_rays + ray_indices[i], isect_indices[i]] = lengths[i]
        for j in range(n_rays):
            g[x*n_rays + j] = intensities[j]
    return [L, g]

def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)
    x = np.dot(np.transpose(L), L)
    a = compute_cholesky(x)
    y = solve_cholesky(a, np.dot(L.T, g))

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))
    tim = np.reshape(y, (n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print(setup_system_tomograph(200,200,0))
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
