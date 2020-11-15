
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/smoke3_mov200_f400')
parser.add_argument("--num_param", type=int, default=2)
parser.add_argument("--path_format", type=str, default='%d_%d.npz')
parser.add_argument("--p0", type=str, default='scenes')
parser.add_argument("--p1", type=str, default='frames')

num_s = 200
num_f = 400
num_sim = num_s*num_f
parser.add_argument("--min_src_pos", type=float, default=0.1)
parser.add_argument("--max_src_pos", type=float, default=0.9)
parser.add_argument("--src_y_pos", type=float, default=0.1)
parser.add_argument("--src_radius", type=float, default=0.08)
parser.add_argument("--min_scenes", type=int, default=0)
parser.add_argument("--max_scenes", type=int, default=num_s-1)
parser.add_argument("--num_scenes", type=int, default=num_s)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=num_f-1)
parser.add_argument("--num_frames", type=int, default=num_f)
parser.add_argument("--num_simulations", type=int, default=num_sim)
parser.add_argument("--num_dof", type=int, default=2)

parser.add_argument("--resolution_x", type=int, default=48) # 96
parser.add_argument("--resolution_y", type=int, default=72) # 128
parser.add_argument("--resolution_z", type=int, default=48) # 1
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=str, default='xXyYzZ') # xXyY
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--adv_order", type=int, default=2)
parser.add_argument("--clamp_mode", type=int, default=2)

parser.add_argument("--nscale", type=float, default=0.01)
parser.add_argument("--nrepeat", type=int, default=1000)
parser.add_argument("--nseed", type=int, default=123)

args = parser.parse_args()
