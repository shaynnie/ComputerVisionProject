import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_pt(pt):
    mean_pt = np.mean(pt, axis=0)
    new_pt = pt - mean_pt
    mean_dist = np.mean(np.sqrt(new_pt[:, 0] ** 2 + new_pt[:, 1] ** 2))
    scale = np.sqrt(2.) / mean_dist
    T = np.array([[scale, 0, -scale * mean_pt[0]], [0, scale, -scale * mean_pt[1]], [0, 0, 1]])
    new_pt = new_pt @ T
    return new_pt, T

def distance(pt1, pt2, F):
    d12 = np.sum((pt1 @ F * pt2) ** 2, axis=1)
    denom12 = np.sum(F @ pt2.T ** 2, axis=0)
    return d12/denom12

def fundamental_matrix(matches):
    """
    type matches: n-dimensional np.array of matching pair with format of (x1, y1, x2, y2)
    rtype: F, res_err
        F: fundamental matrix
        res_err: the residual error (i.e the mean squared distance between points)
    """

    pt1 = np.concatenate((matches[:, 0:2], np.ones((matches.shape[0], 1))), axis=1)
    pt2 = np.concatenate((matches[:, 2:], np.ones((matches.shape[0], 1))), axis=1)

    new_pt1, T1 = normalize_pt(pt1)
    new_pt2, T2 = normalize_pt(pt2)

    x1 = new_pt1[:, 0]
    y1 = new_pt1[:, 1]
    x2 = new_pt2[:, 0]
    y2 = new_pt2[:, 1]

    A = np.ones((matches.shape[0], 9))
    A[:, 0] = x1 * x2
    A[:, 1] = y1 * x2
    A[:, 2] = x2
    A[:, 3] = x1 * y2
    A[:, 4] = y1 * y2
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1

    U, s, V = np.linalg.svd(A, full_matrices=True)
    f = V[-1].reshape((3, 3))

    U, s, V = np.linalg.svd(f, full_matrices=True)
    f = U @ np.diagflat([s[0], s[1], 0]) @ V
    F = T2.T @ f @ T1

    d12 = np.sum(pt2 @ F * pt1, axis=1)
    denom12 = np.sum(F @ pt1.T ** 2, axis=0)
    d21 = np.sum(pt1 @ F.T * pt2, axis=1)
    denom21 = np.sum(F.T @ pt2.T ** 2, axis=0)
    res_err = np.mean((d12 ** 2)/denom12 + (d21 ** 2)/denom21)/2.

    return F, res_err


def find_rotation_translation(E):
    U, s, V = np.linalg.svd(E, full_matrices=True)
    t = U[:, -1];
    Rz_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    possible_R = [U @ Rz_90 @ V, U @ Rz_90.T @ V, -U @ Rz_90 @ V, -U @ Rz_90.T @ V]
    R1, R2 = [R for R in possible_R if np.linalg.det(R) > 0]

    return [R1, R2], [t, -t]


def find_3d_points(matches, P1, P2):
    n_points = matches.shape[0]
    rec_err = 0.
    points_3d = np.zeros((n_points, 3))
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    for (p_i, (x1, y1, x2, y2)) in enumerate(matches):
        
        A[0, :] = np.array([P1[2, 0] * x1 - P1[0, 0], P1[2, 1] * x1 - P1[0, 1], P1[2, 2] * x1 - P1[0, 2]])
        A[1, :] = np.array([P1[2, 0] * y1 - P1[1, 0], P1[2, 1] * y1 - P1[1, 1], P1[2, 2] * y1 - P1[1, 2]])
        A[2, :] = np.array([P2[2, 0] * x2 - P2[0, 0], P2[2, 1] * x2 - P2[0, 1], P2[2, 2] * x2 - P2[0, 2]])
        A[3, :] = np.array([P2[2, 0] * y2 - P2[1, 0], P2[2, 1] * y2 - P2[1, 1], P2[2, 2] * y2 - P2[1, 2]])
        b[0, 0] = P1[0, 3] - P1[2, 3] * x1
        b[1, 0] = P1[1, 3] - P1[2, 3] * y1
        b[2, 0] = P2[0, 3] - P2[2, 3] * x2
        b[3, 0] = P2[1, 3] - P2[2, 3] * y2

        # X, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        X = np.linalg.inv(A.T @ A) @ A.T @ b
        points_3d[p_i, :] = X.reshape((1, 3))

        X = np.append(X, np.array([1.]))        
        denom1 = np.sum(P1[2, :] * X)
        denom2 = np.sum(P2[2, :] * X)
        pX1 = np.sum(P1[0, :] * X) / denom1
        pY1 = np.sum(P1[1, :] * X) / denom1
        pX2 = np.sum(P2[0, :] * X) / denom2
        pY2 = np.sum(P2[1, :] * X) / denom2
        rec_err += np.sqrt(((pX1 - x1) ** 2 + (pY1 - y1) ** 2)/2) + np.sqrt(((pX2 - x2) ** 2 + (pY2 - y2) ** 2)/2)
    
    rec_err = rec_err/(2. * n_points)

    return points_3d, rec_err

def plot_3d(points, t2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, c='r', marker='o', label='camera 1')
    ax.scatter(*t2, c='r', marker='x', label='camera 2')
    for point in points:
        ax.scatter(*point, c='b', marker='o')

    ax.scatter(*points[0, :], c='b', marker='o', label='3D points') #for legend
    ax.legend()
    ax.set_title('3D point cloud of input house')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_2d(points, t2):
    x = -points[:, 0]
    y = -points[:, 1]
    z = points[:, 2]
    
    plt.scatter(x, y, c=z, cmap='gray_r')
    plt.title('Depth Map')

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normalized Depth')
    cbar.ax.tick_params(labelsize=6)
    plt.show()
