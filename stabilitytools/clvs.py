import numpy as np
import scipy


def normalize(M):
    """Normalizes columns of M individually"""
    nM = np.zeros(M.shape)  # normalized matrix
    nV = np.zeros(np.shape(M)[1])  # norms of columns

    for i in range(M.shape[1]):
        nV[i] = scipy.linalg.norm(M[:, i])
        nM[:, i] = M[:, i] / nV[i]

    return nM, nV


def timeseriesdot(x, y, multype):
    tsdot = np.einsum(multype, x, y.T)  # Einstein summation. Index i is time.
    return tsdot


def compute_CLV(QQ, RR, dt):
    """
    Computes the Covariant Lyapunov Vectors (CLVs) using the method of Ginelli et al. (PRL, 2007).

    Parameters
    ----------
    QQ : numpy.ndarray
        Array of shape (N_cells, NLy, tly) containing the timeseries of Gram-Schmidt vectors.
    RR : numpy.ndarray
        Array of shape (NLy, NLy, tly) containing the timeseries of the upper-triangular R matrices from QR decomposition.
    dt : float
        Integration time step for the system.

    Returns
    -------
    V : numpy.ndarray
        Array of shape (N_cells, NLy, tly) containing the Covariant Lyapunov Vectors (CLVs) for each timestep. Each column represents a CLV in physical space.

    Notes
    -----
    - The CLVs are computed in reverse time by iteratively solving triangular systems from the QR decomposition.
    - The method normalizes the vectors at each timestep to avoid numerical instability.
    - Ginelli et al. (PRL, 2007) provides the theoretical foundation for this algorithm.
    """
    n_cells_x2 = QQ.shape[0]
    NLy = QQ.shape[1]
    tly = np.shape(QQ)[-1]

    # coordinates of CLVs in local GS vector basis
    C = np.zeros((NLy, NLy, tly))
    D = np.zeros((NLy, tly))  # diagonal matrix
    # coordinates of CLVs in physical space (each column is a vector)
    V = np.zeros((n_cells_x2, NLy, tly))

    # initialise components to I
    C[:, :, -1] = np.eye(NLy)
    D[:, -1] = np.ones(NLy)
    V[:, :, -1] = np.dot(np.real(QQ[:, :, -1]), C[:, :, -1])

    for i in reversed(range(tly-1)):
        C[:, :, i], D[:, i] = normalize(
            scipy.linalg.solve_triangular(np.real(RR[:, :, i]), C[:, :, i+1]))
        V[:, :, i] = np.dot(np.real(QQ[:, :, i]), C[:, :, i])

    # normalize CLVs before measuring their angles.
    timetot = np.shape(V)[-1]

    for i in range(NLy):
        for t in range(timetot-1):
            V[:, i, t] = V[:, i, t] / np.linalg.norm(V[:, i, t])
    return V


def compute_thetas(V, clv_idx):
    """
    Compute the cosines and angles (in degrees) between subspaces
    defined by the vectors in V for given CLV index pairs.

    Parameters
    ----------
    V : ndarray
        Array of shape (timesteps, subspace_dim, vectors) representing the CLVs (Covariant Lyapunov Vectors).
    clv_idx : list of tuples
        List of index pairs (i, j) indicating which CLV vectors to compare.

    Returns
    -------
    costhetas : ndarray
        Cosines of the angles between the specified CLV vector pairs.
    thetas : ndarray
        Angles (in degrees) between the specified CLV vector pairs.
    """
    costhetas = np.array([np.abs(timeseriesdot(V[:, i, :], V[:, j, :], 'ij,ji->j')) for i, j in clv_idx]).T
    thetas = np.degrees(np.arccos(costhetas))
    return costhetas, thetas
