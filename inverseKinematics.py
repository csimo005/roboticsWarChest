import os
import sys
import argparse

import numpy as np
from functools import reduce

def loadDHParam(fname):
    params = []
    with open(fname, 'r') as f:
        for line in f:
            params.append([float(field) for field in line.split(',')])
    
    params = np.array(params)
    return params

def dhParamTransform(alpha, a, d, theta):
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)

    mat = np.asarray([[c_theta, -c_alpha * s_theta,  s_alpha * s_theta, a * c_theta],
                      [s_theta,  c_alpha * c_theta, -s_alpha * c_theta, a * s_theta],
                      [     0.,            s_alpha,            c_alpha,           d],
                      [     0.,                 0.,                 0.,          1.]])

    return mat

def dhGradient(alpha, a, d, theta):
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    
    mat = np.asarray([[-s_theta, -c_alpha * c_theta,  s_alpha * c_theta, -a * s_theta],
                      [ c_theta, -c_alpha * s_theta,  s_alpha * s_theta,  a * c_theta],
                      [     0.,                  0.,                 0.,           0.],
                      [     0.,                  0.,                 0.,           0.]])

    return mat    

def forwardKinematics(angles, dhParams):
    if angles.size == 0:
        return np.eye(4)
    else:
        return reduce(np.matmul, [dhParamTransform(alpha, a, d, theta + theta_offset)
                                              for (alpha, a, d, theta), theta_offset in zip(dhParams, angles)])

def inverseKinematics(goal, dhParams, initial_solution=None):
    """Attempts to solve the inverse kinematics given a goal transformation and
       dh parameters using the jacobian inverse technique.
    """

    if initial_solution is None:
        solution = np.zeros((dhParams.shape[0],))
    else:
        solution = initial_solution

    it = 0    
    delta = 1
    while delta > 1e-10 and it < 10000:
        currentTransform = forwardKinematics(solution, dhParams)
        error = (goal - currentTransform)/12 #12 values in the transform we are trying to solver for

        update = np.zeros(solution.shape)
        for i in range(len(dhParams)):
            linkTransform = dhParamTransform(*list(dhParams[i]))
            currentTransform = forwardKinematics(solution[i+1:], dhParams[i+1:])
            update[i] = np.mean((error @ currentTransform.T) * dhGradient(*list(dhParams[i])))
            error = linkTransform.T @ error

        delta = np.sqrt(np.sum(update**2))
        solution = solution - 0.01 * update

        it += 1

    return solution

def rotX(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[1., 0., 0.],
                     [0., ct,-st],
                     [0., st, ct]])

def rotY(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ ct, 0., st],
                     [ 0., 1., 0.],
                     [-st, 0., ct]])

def rotZ(theta):
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct,-st, 0.],
                     [st, ct, 0.],
                     [0., 0., 1.]])

def invTransform(tform):
    inv = np.eye(4)
    inv[:3, :3] = tform[:3, :3].T
    inv[:3, 3:] = -tform[:3, :3].T @ tform[:3, 3:]

    return inv

def main(args):
    dhParams = loadDHParam(args.param_file)
    assert len(dhParams) == args.dof

    goalTform = np.eye(4)
    goalTform[:3, :3] = rotZ(args.Y) @ rotY(args.P) @ rotX(args.R)
    goalTform[:3, 3:] = np.array([[args.x, args.y, args.z]]).T

    angles = inverseKinematics(goalTform, dhParams)
    print(('{:.8f} '*len(angles)).format(*list(angles)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str, nargs='?')
    parser.add_argument('dof', type=int, nargs='?')
    parser.add_argument('--x', type=float, default=0., nargs='?')
    parser.add_argument('--y', type=float, default=0., nargs='?')
    parser.add_argument('--z', type=float, default=0., nargs='?')
    parser.add_argument('--R', type=float, default=0., nargs='?')
    parser.add_argument('--P', type=float, default=0., nargs='?')
    parser.add_argument('--Y', type=float, default=0., nargs='?')

    args = parser.parse_args()
    main(args)
