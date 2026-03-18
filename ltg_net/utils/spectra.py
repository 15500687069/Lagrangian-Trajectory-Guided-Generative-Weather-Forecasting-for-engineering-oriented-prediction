from __future__ import annotations

import torch


def radial_power_spectrum_2d(
    field: torch.Tensor,
    max_wavenumber: int | None = None,
) -> torch.Tensor:
    # field: [B, H, W]
    fft = torch.fft.rfft2(field, norm="ortho")
    power = (fft.real**2 + fft.imag**2).mean(dim=0)  # [H, W//2+1]

    h, w2 = power.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=field.device),
        torch.arange(w2, device=field.device),
        indexing="ij",
    )
    ky = torch.minimum(yy, h - yy).float()
    kx = xx.float()
    kr = torch.sqrt(kx**2 + ky**2)
    kr_int = kr.round().long()
    if max_wavenumber is None:
        max_wavenumber = int(kr_int.max().item())
    max_wavenumber = min(max_wavenumber, int(kr_int.max().item()))

    spectrum = torch.zeros(max_wavenumber + 1, device=field.device)
    counts = torch.zeros(max_wavenumber + 1, device=field.device)
    for k in range(max_wavenumber + 1):
        mask = kr_int == k
        if mask.any():
            spectrum[k] = power[mask].mean()
            counts[k] = mask.sum()
    return spectrum


def spectral_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_wavenumber: int = 64,
) -> torch.Tensor:
    # pred/target: [B, C, H, W]
    losses = []
    for c in range(pred.shape[1]):
        sp = radial_power_spectrum_2d(pred[:, c], max_wavenumber=max_wavenumber)
        st = radial_power_spectrum_2d(target[:, c], max_wavenumber=max_wavenumber)
        losses.append(torch.mean(torch.abs(torch.log1p(sp) - torch.log1p(st))))
    return torch.stack(losses).mean()
