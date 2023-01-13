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

def forwardKinematics(angles, dhParams):
    return reduce(np.matmul, [dhParamTransform(alpha, a, d, theta + theta_offset)
                                          for (alpha, a, d, theta), theta_offset in zip(dhParams, angles)])

def main(args):
    dhParams = loadDHParam(args.param_file)
    assert len(dhParams) == args.dof

    if len(args.angles) < args.dof:
        print('Warning: Specified {} degrees of freedom, but only {} joint angles provided'.format(args.dof, len(args.angles)))
        print('Remaining values assumed equal to zero')
        angles = args.angles + [0.] * (args.dof - len(args.angles))
    elif len(args.angles) > args.dof:
        print('Warning: Specified {} degrees of freedom, but {} joint angles provided'.format(args.dof, len(args.angles)))
        print('Extra values will be discarded')
        angles = args.angles[:args.dof]
    else:
        angles = args.angles
    
    tool2base = forwardKinematics(angles, dhParams)
    print(tool2base)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str, nargs='?')
    parser.add_argument('dof', type=int, nargs='?')
    parser.add_argument('angles', type=float, nargs='*')

    args = parser.parse_args()
    main(args)
