""" Utils for model creation """
from typing import Tuple

import torch

def jacobian3(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # x: bdzyx -> bzyxd
    x = x.permute(0,2,3,4,1)

    dudx = x[:, :, :, 1:, 0] - x[:, :, :, :-1, 0]
    dvdx = x[:, :, :, 1:, 1] - x[:, :, :, :-1, 1]
    dwdx = x[:, :, :, 1:, 2] - x[:, :, :, :-1, 2]
    dudy = x[:, :, 1:, :, 0] - x[:, :, :-1, :, 0]
    dvdy = x[:, :, 1:, :, 1] - x[:, :, :-1, :, 1]
    dwdy = x[:, :, 1:, :, 2] - x[:, :, :-1, :, 2]
    dudz = x[:, 1:, :, :, 0] - x[:, :-1, :, :, 0]
    dvdz = x[:, 1:, :, :, 1] - x[:, :-1, :, :, 1]
    dwdz = x[:, 1:, :, :, 2] - x[:, :-1, :, :, 2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:, :, :, -1], dim=3)), dim=3)
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:, :, :, -1], dim=3)), dim=3)
    dwdx = torch.cat((dwdx, torch.unsqueeze(dwdx[:, :, :, -1], dim=3)), dim=3)

    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:, :, -1, :], dim=2)), dim=2)
    dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:, :, -1, :], dim=2)), dim=2)
    dwdy = torch.cat((dwdy, torch.unsqueeze(dwdy[:, :, -1, :], dim=2)), dim=2)

    dudz = torch.cat((dudz, torch.unsqueeze(dudz[:, -1, :, :], dim=1)), dim=1)
    dvdz = torch.cat((dvdz, torch.unsqueeze(dvdz[:, -1, :, :], dim=1)), dim=1)
    dwdz = torch.cat((dwdz, torch.unsqueeze(dwdz[:, -1, :, :], dim=1)), dim=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy

    j = torch.stack([dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz], dim=-1)
    c = torch.stack([u, v, w], dim=-1)

    # j: bzyxd -> bdzyx
    j = j.permute(0,4,1,2,3).contiguous()
    c = c.permute(0,4,1,2,3).contiguous()

    return j, c
