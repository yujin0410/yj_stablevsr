import torch
import torch.nn.functional as F
import numpy as np

def warp_dtcwt_high_bands(Yh_list, flow):
    Yh_L1 = Yh_list[0]
    B, C, N_dir, H_yh, W_yh, _ = Yh_L1.shape

    flow_resized = F.interpolate(flow, size=(H_yh, W_yh), mode='bilinear', align_corners=False)
    flow_resized[:, 0] *= (W_yh / flow.shape[3])
    flow_resized[:, 1] *= (H_yh / flow.shape[2])
    u = flow_resized[:, 0:1]
    v = flow_resized[:, 1:2]

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H_yh, device=flow.device),
        torch.linspace(-1, 1, W_yh, device=flow.device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    norm_u = u.squeeze(1) / (W_yh / 2)
    norm_v = v.squeeze(1) / (H_yh / 2)

    sample_x = grid_x + norm_u  # [B, H, W]
    sample_y = grid_y + norm_v
    grid = torch.stack([sample_x, sample_y], dim=-1)  # [B, H, W, 2]

    w_high, w_low = 0.75 * np.pi, 0.25 * np.pi
    center_freqs = {
        0: (w_high, w_low),  1: (w_high, w_high),
        2: (w_low,  w_high), 3: (-w_low,  w_high),
        4: (-w_high, w_high),5: (-w_high, w_low)
    }

    warped_coeffs = []
    for d in range(6):
        wx, wy = center_freqs[d]
        # u, v are in subband pixels (level 1: 1 subband px = 2 original px)
        # DTCWT steering property requires displacement in original pixels: d = 2 * u_subband
        delta_phi = -(wx * 2 * u + wy * 2 * v).to(dtype=Yh_L1.dtype)
        cos_d = torch.cos(delta_phi)  # [B, 1, H, W]
        sin_d = torch.sin(delta_phi)

        Re = Yh_L1[:, :, d, :, :, 0]  # [B, C, H, W]
        Im = Yh_L1[:, :, d, :, :, 1]

        Re_flat = Re.reshape(B * C, 1, H_yh, W_yh)
        Im_flat = Im.reshape(B * C, 1, H_yh, W_yh)
        grid_exp = grid.unsqueeze(1).expand(-1, C, -1, -1, -1).reshape(B * C, H_yh, W_yh, 2)

        Re_warped_spatial = F.grid_sample(Re_flat, grid_exp, 
                                           mode='bilinear', 
                                           padding_mode='border',
                                           align_corners=False).reshape(B, C, H_yh, W_yh)
        Im_warped_spatial = F.grid_sample(Im_flat, grid_exp,
                                           mode='bilinear',
                                           padding_mode='border', 
                                           align_corners=False).reshape(B, C, H_yh, W_yh)

        Re_out = Re_warped_spatial * cos_d - Im_warped_spatial * sin_d
        Im_out = Re_warped_spatial * sin_d + Im_warped_spatial * cos_d

        warped_coeffs.append(torch.stack([Re_out, Im_out], dim=-1))

    return torch.stack(warped_coeffs, dim=2)
