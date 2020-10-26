from pathlib import Path

import numpy as np


import cv2

d = Path("log/smoke_pos21_size5_f200/1024_201513_de_tag/10_2")
#d = Path("data/smoke_pos21_size5_f200/v/")
out_directory = Path("out/")
out_directory.mkdir(exist_ok=True)

for i in range(200):

    data_fn = d / f"{i}.npz"
    #data_fn = d / f"10_2_{i}.npz"
    data = np.load(data_fn)
    print(list(data.keys()))
    img = data['x']

    img = np.concatenate([img, np.zeros(shape=list(img.shape[:2]) + [1])], axis=2)
    print(f'img shape:{np.linalg.norm(img)}')
    img *= 100
    img = img.astype('uint8')

    img_fn = out_directory / f"img{i:04}.jpg"
    img_fn = str(img_fn)
    cv2.imwrite(img_fn, img)


