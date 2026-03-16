import torch
import torch.nn.functional as F
import numpy as np

def warp_dtcwt_high_bands(Yh_list, flow):
    Yh_L1 = Yh_list[0]
    B, C, N_dir, H_yh, W_yh, _ = Yh_L1.shape

    # 1. Flow 리사이즈 + 서브밴드 픽셀 단위로 스케일 보정
    flow_resized = F.interpolate(flow, size=(H_yh, W_yh), mode='bilinear', align_corners=False)
    flow_resized[:, 0] *= (W_yh / flow.shape[3])
    flow_resized[:, 1] *= (H_yh / flow.shape[2])
    u = flow_resized[:, 0:1]
    v = flow_resized[:, 1:2]

    # 2. grid_sample용 normalized grid 생성
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

    # 3. 절대 픽셀 좌표계 생성 (demodulation/remodulation 공용)
    idx_y, idx_x = torch.meshgrid(
        torch.arange(H_yh, device=flow.device, dtype=flow.dtype),
        torch.arange(W_yh, device=flow.device, dtype=flow.dtype),
        indexing='ij'
    )
    idx_x = idx_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    idx_y = idx_y.unsqueeze(0).unsqueeze(0)

    w_high, w_low = 0.75 * np.pi, 0.25 * np.pi
    center_freqs = {
        0: (w_high, w_low),  1: (w_high, w_high),
        2: (w_low,  w_high), 3: (-w_low,  w_high),
        4: (-w_high, w_high),5: (-w_high, w_low)
    }

    warped_coeffs = []
    for d in range(6):
        wx, wy = center_freqs[d]

        Re = Yh_L1[:, :, d, :, :, 0]  # [B, C, H, W]
        Im = Yh_L1[:, :, d, :, :, 1]

        # STEP 1: Demodulation — 고주파를 기저대역(baseband)으로 변환
        # Y_BB = Y * e^{-j*(wx*x + wy*y)}
        phi_grid = (wx * idx_x + wy * idx_y).to(dtype=Re.dtype)
        cos_phi = torch.cos(phi_grid)
        sin_phi = torch.sin(phi_grid)

        Re_BB = Re * cos_phi + Im * sin_phi
        Im_BB = Im * cos_phi - Re * sin_phi

        # STEP 2: Spatial warp — 기저대역에서 안전하게 워핑 (진동 없음)
        Re_BB_flat = Re_BB.reshape(B * C, 1, H_yh, W_yh)
        Im_BB_flat = Im_BB.reshape(B * C, 1, H_yh, W_yh)
        grid_exp = grid.unsqueeze(1).expand(-1, C, -1, -1, -1).reshape(B * C, H_yh, W_yh, 2)

        Re_BB_warped = F.grid_sample(Re_BB_flat, grid_exp,
                                     mode='bilinear', padding_mode='border',
                                     align_corners=False).reshape(B, C, H_yh, W_yh)
        Im_BB_warped = F.grid_sample(Im_BB_flat, grid_exp,
                                     mode='bilinear', padding_mode='border',
                                     align_corners=False).reshape(B, C, H_yh, W_yh)

        # STEP 3: Remodulation — 목적지 좌표(phi_grid) 기준으로 원래 주파수 복원
        # Y_out = Y_BB_warped * e^{+j*(wx*x + wy*y)}  ← 소스 좌표 아님, 목적지 좌표!
        # 결과: Y_out(x,y) = Y_src(x+u, y+v) * e^{-j*(wx*u + wy*v)}  (스티어링 공식 일치)
        Re_out = Re_BB_warped * cos_phi - Im_BB_warped * sin_phi
        Im_out = Re_BB_warped * sin_phi + Im_BB_warped * cos_phi

        warped_coeffs.append(torch.stack([Re_out, Im_out], dim=-1))

    return torch.stack(warped_coeffs, dim=2)
