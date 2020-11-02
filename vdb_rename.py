""" A script to rename a sequnce of vdb files """
from pathlib import Path
import os
import shutil

input_directory = Path("data/smoke3_vel5_buo3_f250/vdb/")
output_directory = Path("out")
if output_directory.exists():
    shutil.rmtree(output_directory)

output_directory.mkdir()

fns = list(input_directory.glob("0_0_*.vdb"))
fns.sort(key=os.path.getmtime)

print(f'fns: {fns}')

for i, fn_in in enumerate(fns):
    fn_out = output_directory / f"{i:04}.vdb"
    shutil.copyfile(fn_in, fn_out)
