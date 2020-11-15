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

from collections import deque
from perlin import TileableNoise

from utils.smoke3_mov import args


def advect(v_path: Path):
    def get_param(p1):
        min_p1 = args.min_scenes
        max_p1 = args.max_scenes
        num_p1 = args.num_scenes
        p1_ = p1 / (num_p1 - 1) * (max_p1 - min_p1) + min_p1
        return p1_

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

    dim = 2
    if res_z > 1: dim = 3
    s = Solver(name='main', gridSize=gs, dim=dim)
    s.timestep = args.time_step

    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)

    flags.initDomain(boundaryWidth=args.bWidth)
    flags.fillGrid()
    setOpenBound(flags, args.bWidth, args.open_bound, FlagOutflow | FlagEmpty)

    vel.clear()
    density.clear()

    radius = gs.x * args.src_radius

    if (GUI):
        gui = Gui()
        gui.show(True)
        gui.nextVec3Display()
        gui.nextVec3Display()
        gui.nextVec3Display()
    # gui.pause()

    if res_z > 1:
        d_ = np.zeros([res_z, res_y, res_x], dtype=np.float32)
    else:
        d_ = np.zeros([res_y, res_x], dtype=np.float32)
    for t in trange(args.num_frames):
        v_path_ = v_path / f"{t}.npz"
        v_path_ = str(v_path_)
        with np.load(v_path_) as data:
            v = data['x']
            if res_z == 1:
                v = np.dstack((v, np.zeros([res_y, res_x, 1])))
            p = data['y']

        copyArrayToGridMAC(v, vel)

        nx = p[0, -1]
        nz = 0.5
        if res_z > 1: nz = p[1, -1]
        source = s.create(Sphere, center=gs * vec3(nx, args.src_y_pos, nz), radius=radius)
        source.applyToGrid(grid=density, value=1)
        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=args.adv_order,
                           openBounds=True, boundaryWidth=args.bWidth, clampMode=args.clamp_mode)

        copyGridToArrayReal(density, d_)

        img_path = os.path.join(img_dir, '%04d.png' % t)
        if res_z > 1:
            d_img = np.mean(d_[:, ::-1], axis=0) * 255
        else:
            d_img = d_[::-1] * 255
        d_img = np.stack((d_img, d_img, d_img), axis=-1).astype(np.uint8)
        d_img = Image.fromarray(d_img)
        d_img.save(img_path)

        # save as vdb file
        vdb_file_path = os.path.join(vdb_dir, f'{t:04}.vdb')
        density.save(vdb_file_path)

        s.step()


if __name__ == '__main__':
    velocity_field = Path("/Users/vivanov/Projects/deep-fluids/experiments/Autoencoder/velocity_field/")
    advect(velocity_field)
