import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d(0)
    base_functions = []

    # TODO: Generate Lagrange base polynomials and interpolation polynomial
    for i in range(0, x.size):
        tmp = 1.0
        for j in range(0, x.size):
            if i != j:
                tmp = tmp * np.poly1d([1, -x[j]])
                a = x[i] - x[j]
                tmp = tmp / a
        base_functions.append(tmp)
    for i in range(0, x.size):
        polynomial = polynomial + base_functions[i] * y[i]

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials
    for i in range(0, x.size - 1):
        Matrix_M = np.array([
            [1, x[i], x[i] ** 2, x[i] ** 3],
            [0, 1, 2 * x[i], 3 * x[i] ** 2],
            [1, x[i + 1], x[i + 1] ** 2, x[i + 1] ** 3],
            [0, 1, 2 * x[i + 1], 3 * x[i + 1] ** 2]
        ])
        b = [y[i], yp[i], y[i + 1], yp[i + 1]]
        A = np.linalg.solve(Matrix_M, b)
        spline.append(np.poly1d(A[::-1]))
    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions
    A = np.zeros((4 * (x.size - 1), 4 * (x.size - 1)), dtype=np.float64)

    # TODO solve linear system for the coefficients of the spline
    b = np.zeros((4 * (x.size - 1)))

    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(0, x.size - 2):
        A[4 * i, 4 * i] = x[i] ** 3
        A[4 * i, 4 * i + 1] = x[i] ** 2
        A[4 * i, 4 * i + 2] = x[i]
        A[4 * i, 4 * i + 3] = 1

        A[4 * i + 1, 4 * i] = x[i + 1] ** 3
        A[4 * i + 1, 4 * i + 1] = x[i + 1] ** 2
        A[4 * i + 1, 4 * i + 2] = x[i + 1]
        A[4 * i + 1, 4 * i + 3] = 1

        A[4 * i + 2, 4 * i] = 3 * x[i + 1] ** 2
        A[4 * i + 2, 4 * i + 1] = 2 * x[i + 1]
        A[4 * i + 2, 4 * i + 2] = 1
        A[4 * i + 2, 4 * i + 4] = -3 * x[i + 1] ** 2
        A[4 * i + 2, 4 * i + 5] = -2 * x[i + 1]
        A[4 * i + 2, 4 * i + 6] = -1

        A[4 * i + 3, 4 * i] = 6 * x[i + 1]
        A[4 * i + 3, 4 * i + 1] = 2
        A[4 * i + 3, 4 * i + 4] = -6 * x[i + 1]
        A[4 * i + 3, 4 * i + 5] = -2

    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 4] = x[x.size - 2] ** 3
    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 3] = x[x.size - 2] ** 2
    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 2] = x[x.size - 2]
    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 1] = 1

    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 4] = x[x.size - 1] ** 3
    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 3] = x[x.size - 1] ** 2
    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 2] = x[x.size - 1]
    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 1] = 1

    A[4 * (x.size - 1) - 2, 0] = 6 * x[0]
    A[4 * (x.size - 1) - 2, 1] = 2

    A[4 * (x.size - 1) - 1, 4 * (x.size - 1) - 4] = 6 * x[x.size - 1]
    A[4 * (x.size - 1) - 1, 4 * (x.size - 1) - 3] = 2

    for i in range(0, x.size - 1):
        b[4 * i] = y[i]
        b[4 * i + 1] = y[i + 1]

    C = np.linalg.solve(A, b)

    spline = []
    for i in range(0, x.size -1 ):
        spline.append(np.poly1d(C[4 * i:4 * i + 4]))

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions
    A = np.zeros((4 * (x.size - 1), 4 * (x.size - 1)), dtype=np.float64)

    # TODO solve linear system for the coefficients of the spline
    b = np.zeros((4 * (x.size - 1)))
    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(0, x.size - 2):
        A[4 * i, 4 * i] = x[i] ** 3
        A[4 * i, 4 * i + 1] = x[i] ** 2
        A[4 * i, 4 * i + 2] = x[i]
        A[4 * i, 4 * i + 3] = 1

        A[4 * i + 1, 4 * i] = x[i + 1] ** 3
        A[4 * i + 1, 4 * i + 1] = x[i + 1] ** 2
        A[4 * i + 1, 4 * i + 2] = x[i + 1]
        A[4 * i + 1, 4 * i + 3] = 1

        A[4 * i + 2, 4 * i] = 3 * x[i + 1] ** 2
        A[4 * i + 2, 4 * i + 1] = 2 * x[i + 1]
        A[4 * i + 2, 4 * i + 2] = 1
        A[4 * i + 2, 4 * i + 4] = -3 * x[i + 1] ** 2
        A[4 * i + 2, 4 * i + 5] = -2 * x[i + 1]
        A[4 * i + 2, 4 * i + 6] = -1

        A[4 * i + 3, 4 * i] = 6 * x[i + 1]
        A[4 * i + 3, 4 * i + 1] = 2
        A[4 * i + 3, 4 * i + 4] = -6 * x[i + 1]
        A[4 * i + 3, 4 * i + 5] = -2

    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 4] = x[x.size - 2] ** 3
    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 3] = x[x.size - 2] ** 2
    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 2] = x[x.size - 2]
    A[4 * (x.size - 1) - 4, 4 * (x.size - 1) - 1] = 1

    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 4] = x[x.size - 1] ** 3
    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 3] = x[x.size - 1] ** 2
    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 2] = x[x.size - 1]
    A[4 * (x.size - 1) - 3, 4 * (x.size - 1) - 1] = 1

    A[4 * (x.size - 1) - 2, 0] = 6 * x[0]
    A[4 * (x.size - 1) - 2, 1] = 2
    A[4 * (x.size - 1) - 1, 0] = 3 * x[0] ** 2
    A[4 * (x.size - 1) - 1, 1] = 2 * x[0]
    A[4 * (x.size - 1) - 1, 2] = 1

    A[4 * (x.size - 1) - 2, 4 * (x.size - 1) - 4] = -6 * x[x.size - 1]
    A[4 * (x.size - 1) - 2, 4 * (x.size - 1) - 3] = -2
    A[4 * (x.size - 1) - 1, 4 * (x.size - 1) - 4] = -3 * x[x.size - 1] ** 2
    A[4 * (x.size - 1) - 1, 4 * (x.size - 1) - 3] = -2 * x[x.size - 1]
    A[4 * (x.size - 1) - 1, 4 * (x.size - 1) - 2] = -1

    for i in range(x.size - 1):
        b[4 * i] = y[i]
        b[4 * i + 1] = y[i + 1]
    C = np.linalg.solve(A, b)

    spline = []
    for i in range(0, x.size - 1):
        spline.append(np.poly1d(C[4 * i:4 * i + 4]))

    return spline


if __name__ == '__main__':

    x = np.array( [1.0, 2.0, 3.0, 4.0])
    y = np.array( [3.0, 2.0, 4.0, 1.0])

    splines = natural_cubic_interpolation( x, y)

    # # x-values to be interpolated
    # keytimes = np.linspace(0, 200, 11)
    # # y-values to be interpolated
    # keyframes = [np.array([0., -0.05, -0.2, -0.2, 0.2, -0.2, 0.25, -0.3, 0.3, 0.1, 0.2]),
    #              np.array([0., 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.3, 0.3])] * 5
    # keyframes.append(keyframes[0])
    # splines = []
    # for i in range(11):  # Iterate over all animated parts
    #     x = keytimes
    #     y = np.array([keyframes[k][i] for k in range(11)])
    #     spline = natural_cubic_interpolation(x, y)
    #     if len(spline) == 0:
    #         animate(keytimes, keyframes, linear_animation(keytimes, keyframes))
    #         self.fail("Natural cubic interpolation not implemented.")
    #     splines.append(spline)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
