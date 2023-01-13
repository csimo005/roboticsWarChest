import os
import sys
import argparse

import numpy as np

def rpyToQuaternion(roll, pitch, yaw):
    c_r, s_r = np.cos(roll/2), np.sin(roll/2)
    c_p, s_p = np.cos(pitch/2), np.sin(pitch/2)
    c_y, s_y = np.cos(yaw/2), np.sin(yaw/2)

    w = c_r*c_p*c_y + s_r*s_p*s_y
    x = s_r*c_p*c_y - c_r*s_p*s_y
    y = c_r*s_p*c_y + s_r*c_p*s_y
    z = c_r*c_p*s_y - s_r*s_p*c_y

    return w, x, y, z

def quaternionToRPY(w, x, y, z):
    r = np.arctan2(2*(w*x + y*z), 1-2*(x**2 + y**2))
    p = -np.pi/2 + 2*np.arctan2(np.sqrt(1+2*(w*y - x*z)), np.sqrt(1-2*(x*y - x*z)))
    y = np.arctan2(2*(w*z + x*y), 1-2*(y**2 + z**2))
    return r, p, y

rpy_format = 'RPY Format:\nroll = {:.6f}\npitch = {:.6f}\nyaw = {:.6f}'
quaternion_format = 'Quaternion Format:\nw = {:.6f}\nx = {:.6f}\ny = {:.6f}\nz = {:.6f}'

def main(args):
    if args.format == 'rpy':
        assert len(args.values) == 3
        print(quaternion_format.format(*rpyToQuaternion(*args.values)))
    elif args.format == 'quaternion':
        assert len(args.values) == 4
        print(rpy_format.format(*quaternionToRPY(*args.values)))
    else:
        raise ValueError('Unknown rotation format {}'.format(args.format))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('format', type=str, choices=['rpy', 'quaternion'])
    parser.add_argument('values', type=float, nargs='*')

    args = parser.parse_args()
    main(args)
