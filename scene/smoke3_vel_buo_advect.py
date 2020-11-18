import argparse
import shutil
from datetime import datetime
import os
from pathlib import Path

from tqdm import trange
import numpy as np
from PIL import Image
import gc

try:
    from manta import *
except ImportError:
    pass

from utils.smoke3_vel_buo import args


def advect():
    # def get_param(p1, p2):
    #     min_p1 = args.min_inflow
    #     max_p1 = args.max_inflow
    #     num_p1 = args.num_inflow
    #     min_p2 = args.min_buoyancy
    #     max_p2 = args.max_buoyancy
    #     num_p2 = args.num_buoyancy
    #     p1_ = p1/(num_p1-1) * (max_p1-min_p1) + min_p1
    #     p2_ = p2/(num_p2-1) * (max_p2-min_p2) + min_p2
    #     return p1_, p2_

    p1, p2 = 0, 0
    # p1_, p2_ = get_param(p1, p2)
    v_path = os.path.join(args.log_dir, 'v')
    img_dir = os.path.join(args.log_dir, 'd_adv')
    vdb_dir = os.path.join(args.log_dir, 'd_vdb')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if not os.path.exists(vdb_dir):
        os.makedirs(vdb_dir)

    # solver params
    res_x = args.resolution_x
    res_y = args.resolution_y
    res_z = args.resolution_z
    gs = vec3(res_x, res_y, res_z)

    s = Solver(name='main', gridSize=gs, dim=3)
    s.frameLength = 1.0
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)

    # noise field, tweak a bit for smoke source
    noise = s.create(NoiseField, loadFromFile=True)
    noise.posScale = vec3(45)
    noise.clamp = True
    noise.clampNeg = 0
    noise.clampPos = 1
    noise.valOffset = 0.75
    noise.timeAnim = 0.2

    flags.initDomain(boundaryWidth=args.bWidth)
    flags.fillGrid()
    setOpenBound(flags, args.bWidth, args.open_bound, FlagOutflow | FlagEmpty)

    src_center = gs * vec3(args.src_x_pos, args.src_y_pos, args.src_z_pos)
    src_radius = args.resolution_y * args.src_radius
    src_z = gs * vec3(0, args.src_height, 0)
    source = s.create(Cylinder, center=src_center, radius=src_radius, z=src_z)

    if (GUI):
        gui = Gui()
        gui.show(True)
        gui.nextVec3Display()
        gui.nextVec3Display()
        gui.nextVec3Display()
        gui.pause()

    d_ = np.zeros([res_z, res_y, res_x], dtype=np.float32)
    for t in trange(args.num_frames):
        v_path_ = f"/home/vivanov/Projects/deep-fluids/experiments/Generator/velocity_field/{t}.npz"

        with np.load(v_path_) as data:
            v = data['x']

        copyArrayToGridMAC(v, vel)
        densityInflow(flags=flags, density=density, noise=noise, shape=source, scale=1, sigma=0.5)
        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
                           openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)
        copyGridToArrayReal(density, d_)

        # save as png file
        d_file_path = os.path.join(img_dir, '%04d.png' % t)
        d_img = np.mean(d_[:, ::-1], axis=0) * 255  # yx
        d_img = np.stack((d_img, d_img, d_img), axis=-1).astype(np.uint8)
        d_img = Image.fromarray(d_img)
        d_img.save(d_file_path)

        # save as vdb file
        vdb_file_path = os.path.join(vdb_dir, f'{t:04}.vdb')
        density.save(vdb_file_path)

        s.step()


if __name__ == '__main__':
    advect()
