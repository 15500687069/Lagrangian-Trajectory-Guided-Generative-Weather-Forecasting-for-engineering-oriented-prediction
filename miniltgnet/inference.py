from __future__ import annotations

from typing import Any

import torch


def _wrap_lon_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b + 180.0) % 360.0 - 180.0


def _clamp_geo(state: torch.Tensor) -> torch.Tensor:
    lat = state[..., 0].clamp(-90.0, 90.0)
    lon = state[..., 1] % 360.0
    return torch.stack([lat, lon], dim=-1)


def _blend_geo(base: torch.Tensor, ref: torch.Tensor, weight_base: torch.Tensor | float) -> torch.Tensor:
    # weight_base=1 keeps base, weight_base=0 keeps ref
    if not torch.is_tensor(weight_base):
        w = torch.tensor(float(weight_base), device=base.device, dtype=base.dtype)
    else:
        w = weight_base.to(device=base.device, dtype=base.dtype)
    while w.ndim < (base.ndim - 1):
        w = w.unsqueeze(-1)
    lat = w * base[..., 0] + (1.0 - w) * ref[..., 0]
    dlon = _wrap_lon_diff(base[..., 1], ref[..., 1])
    lon = (ref[..., 1] + w * dlon) % 360.0
    return _clamp_geo(torch.stack([lat, lon], dim=-1))


def _predict_persistence_traj(traj_hist: torch.Tensor, horizon: int) -> torch.Tensor:
    last = traj_hist[:, -1].float()
    return last.unsqueeze(1).repeat(1, int(horizon), 1, 1)


def _predict_linear_traj(traj_hist: torch.Tensor, horizon: int) -> torch.Tensor:
    if int(traj_hist.shape[1]) < 2:
        return _predict_persistence_traj(traj_hist, horizon)
    prev = traj_hist[:, -1].float()
    delta_lat = traj_hist[:, -1, :, 0] - traj_hist[:, -2, :, 0]
    delta_lon = _wrap_lon_diff(traj_hist[:, -1, :, 1], traj_hist[:, -2, :, 1])
    outs: list[torch.Tensor] = []
    for _ in range(int(horizon)):
        lat = prev[..., 0] + delta_lat
        lon = prev[..., 1] + delta_lon
        prev = _clamp_geo(torch.stack([lat, lon], dim=-1))
        outs.append(prev)
    return torch.stack(outs, dim=1)


def _parse_post_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    ecfg = cfg.get("evaluation", cfg)
    pcfg = ecfg.get("inference_postprocess", {})
    return pcfg if isinstance(pcfg, dict) else {}


def _traj_postprocess(
    pred_traj: torch.Tensor,
    traj_hist: torch.Tensor,
    post_cfg: dict[str, Any],
) -> torch.Tensor:
    if not bool(post_cfg.get("traj_enabled", True)):
        return pred_traj

    ref_mode = str(post_cfg.get("traj_ref_mode", "mixed")).lower()
    linear_weight = float(post_cfg.get("traj_blend_linear", 0.30))
    persistence_weight = float(post_cfg.get("traj_blend_persistence", 0.10))
    lead_boost = float(post_cfg.get("traj_lead_persistence_boost", 0.55))
    conf_power = float(post_cfg.get("traj_confidence_power", 1.4))
    global_pers_weight = float(post_cfg.get("traj_global_persistence_weight", 0.0))
    prefix_leads = int(post_cfg.get("traj_persistence_prefix_leads", 0))
    prefix_pers_weight = float(post_cfg.get("traj_prefix_persistence_weight", 1.0))
    speed_sigma = float(post_cfg.get("traj_speed_sigma", 0.70))
    min_model_weight = float(post_cfg.get("traj_min_model_weight", 0.35))
    max_corr = float(post_cfg.get("traj_max_correction_deg", 1.50))

    linear_weight = min(max(linear_weight, 0.0), 1.0)
    persistence_weight = min(max(persistence_weight, 0.0), 1.0)
    lead_boost = min(max(lead_boost, 0.0), 1.0)
    conf_power = max(0.5, conf_power)
    global_pers_weight = min(max(global_pers_weight, 0.0), 1.0)
    prefix_pers_weight = min(max(prefix_pers_weight, 0.0), 1.0)
    min_model_weight = min(max(min_model_weight, 0.0), 1.0)
    speed_sigma = max(1e-4, speed_sigma)

    b, t, o, _ = pred_traj.shape
    traj_hist = traj_hist.float()
    pred_traj = pred_traj.float()

    pers = _predict_persistence_traj(traj_hist, t)
    if ref_mode == "persistence":
        ref = pers
    else:
        lin = _predict_linear_traj(traj_hist, t)
        ref = _blend_geo(base=lin, ref=pers, weight_base=1.0 - persistence_weight)

    if int(traj_hist.shape[1]) >= 2:
        hist_dlat = traj_hist[:, 1:, :, 0] - traj_hist[:, :-1, :, 0]
        hist_dlon = _wrap_lon_diff(traj_hist[:, 1:, :, 1], traj_hist[:, :-1, :, 1])
        hist_speed = torch.sqrt(hist_dlat.square() + hist_dlon.square())
        hist_scale = hist_speed.median(dim=1, keepdim=True).values.clamp_min(1e-4)  # [B,1,O]
    else:
        hist_scale = torch.full((b, 1, o), 0.5, device=pred_traj.device, dtype=pred_traj.dtype)

    step0_lat = pred_traj[:, 0, :, 0] - traj_hist[:, -1, :, 0]
    step0_lon = _wrap_lon_diff(pred_traj[:, 0, :, 1], traj_hist[:, -1, :, 1])
    if t > 1:
        step_lat = pred_traj[:, 1:, :, 0] - pred_traj[:, :-1, :, 0]
        step_lon = _wrap_lon_diff(pred_traj[:, 1:, :, 1], pred_traj[:, :-1, :, 1])
        step_lat = torch.cat([step0_lat.unsqueeze(1), step_lat], dim=1)
        step_lon = torch.cat([step0_lon.unsqueeze(1), step_lon], dim=1)
    else:
        step_lat = step0_lat.unsqueeze(1)
        step_lon = step0_lon.unsqueeze(1)
    pred_speed = torch.sqrt(step_lat.square() + step_lon.square()).clamp_min(1e-6)

    ratio = pred_speed / hist_scale
    dev = torch.abs(torch.log(ratio + 1e-6))
    confidence = torch.exp(-((dev / speed_sigma) ** 2)).clamp(0.0, 1.0)

    lead_frac = torch.linspace(0.0, 1.0, t, device=pred_traj.device, dtype=pred_traj.dtype).view(1, t, 1)
    conf = confidence.clamp(0.0, 1.0).pow(conf_power)
    w_model = 1.0 - linear_weight * (1.0 - conf)
    w_model = w_model * (1.0 - lead_boost * lead_frac)
    w_model = w_model.clamp(min=min_model_weight, max=1.0)
    corrected = _blend_geo(base=pred_traj, ref=ref, weight_base=w_model)
    if global_pers_weight > 0:
        corrected = _blend_geo(base=corrected, ref=pers, weight_base=1.0 - global_pers_weight)

    if max_corr > 0:
        dlat = corrected[..., 0] - pred_traj[..., 0]
        dlon = _wrap_lon_diff(corrected[..., 1], pred_traj[..., 1])
        dnorm = torch.sqrt(dlat.square() + dlon.square()).clamp_min(1e-6)
        scale = torch.minimum(torch.ones_like(dnorm), torch.full_like(dnorm, max_corr) / dnorm)
        lat = pred_traj[..., 0] + scale * dlat
        lon = pred_traj[..., 1] + scale * dlon
        corrected = _clamp_geo(torch.stack([lat, lon], dim=-1))

    if prefix_leads > 0 and prefix_pers_weight > 0:
        lp = min(int(t), int(prefix_leads))
        corrected[:, :lp] = _blend_geo(
            base=corrected[:, :lp],
            ref=pers[:, :lp],
            weight_base=1.0 - prefix_pers_weight,
        )

    return corrected


def _field_spectral_postprocess(
    pred_field: torch.Tensor,
    x_hist: torch.Tensor,
    post_cfg: dict[str, Any],
) -> torch.Tensor:
    if not bool(post_cfg.get("spectral_enabled", True)):
        return pred_field

    hf_blend = float(post_cfg.get("spectral_hf_blend", 0.16))
    mid_blend = float(post_cfg.get("spectral_mid_blend", 0.06))
    lead_boost = float(post_cfg.get("spectral_lead_boost", 0.18))
    k_ratio = float(post_cfg.get("spectral_k_ratio", 0.45))
    gain_strength = float(post_cfg.get("spectral_gain_strength", 0.25))
    power_match = float(post_cfg.get("spectral_power_match", 0.30))
    global_pers_weight = float(post_cfg.get("spectral_global_persistence_weight", 0.0))
    gain_min_k = int(post_cfg.get("spectral_gain_min_k", 2))
    gain_clip_min = float(post_cfg.get("spectral_gain_clip_min", 0.85))
    gain_clip_max = float(post_cfg.get("spectral_gain_clip_max", 1.15))

    hf_blend = min(max(hf_blend, 0.0), 1.0)
    mid_blend = min(max(mid_blend, 0.0), 1.0)
    lead_boost = min(max(lead_boost, 0.0), 1.0)
    k_ratio = min(max(k_ratio, 0.05), 0.95)
    gain_strength = min(max(gain_strength, 0.0), 1.0)
    power_match = min(max(power_match, 0.0), 1.0)
    global_pers_weight = min(max(global_pers_weight, 0.0), 1.0)
    gain_clip_min = max(1e-3, gain_clip_min)
    gain_clip_max = max(gain_clip_min, gain_clip_max)

    squeeze_t = False
    pf = pred_field.float()
    if pf.ndim == 4:
        pf = pf.unsqueeze(1)
        squeeze_t = True
    if pf.ndim != 5 or x_hist.ndim != 5:
        return pred_field

    b, t, c, h, w = pf.shape
    ref = x_hist[:, -1].float().unsqueeze(1).expand(b, t, c, h, w)
    pred4 = pf.reshape(b * t, c, h, w)
    ref4 = ref.reshape(b * t, c, h, w)

    pred_f = torch.fft.rfft2(pred4, dim=(-2, -1), norm="ortho")
    ref_f = torch.fft.rfft2(ref4, dim=(-2, -1), norm="ortho")
    w2 = pred_f.shape[-1]

    yy, xx = torch.meshgrid(
        torch.arange(h, device=pred_f.device),
        torch.arange(w2, device=pred_f.device),
        indexing="ij",
    )
    ky_raw = torch.minimum(yy, h - yy).float()
    kx_raw = xx.float()
    kr_raw = torch.sqrt(kx_raw.square() + ky_raw.square())

    ky = ky_raw / max(1.0, float(h // 2))
    kx = kx_raw / max(1.0, float(w2 - 1))
    radius = torch.sqrt(kx.square() + ky.square())
    high_mask = (radius >= k_ratio)
    mid_mask = (radius >= (k_ratio * 0.6)) & (radius < k_ratio)

    lead_frac = torch.linspace(0.0, 1.0, t, device=pred_f.device, dtype=pred4.dtype)
    lead_idx = torch.arange(t, device=pred_f.device).repeat(b)
    blend_high_bt = (hf_blend + lead_boost * lead_frac[lead_idx]).clamp(0.0, 1.0)
    blend_mid_bt = (mid_blend + 0.5 * lead_boost * lead_frac[lead_idx]).clamp(0.0, 1.0)

    if hf_blend > 0 or lead_boost > 0:
        pred_f = pred_f + blend_high_bt[:, None, None, None] * high_mask[None, None] * (ref_f - pred_f)
    if mid_blend > 0:
        pred_f = pred_f + blend_mid_bt[:, None, None, None] * mid_mask[None, None] * (ref_f - pred_f)

    if (gain_strength > 0 or power_match > 0) and high_mask.any():
        gain_mask = high_mask & (kr_raw >= float(max(0, gain_min_k)))
        if gain_mask.any():
            pwr_pred = pred_f.real.square() + pred_f.imag.square()
            pwr_ref = ref_f.real.square() + ref_f.imag.square()
            p_mean = pwr_pred[..., gain_mask].mean(dim=-1, keepdim=True)
            r_mean = pwr_ref[..., gain_mask].mean(dim=-1, keepdim=True)
            gain = torch.sqrt((r_mean + 1e-6) / (p_mean + 1e-6))
            gain = gain.clamp(min=gain_clip_min, max=gain_clip_max)
            lead_w = lead_frac[lead_idx].view(-1, 1, 1)
            gain = 1.0 + (gain_strength + power_match * lead_w) * (gain - 1.0)
            pred_f[..., gain_mask] = pred_f[..., gain_mask] * gain

    out = torch.fft.irfft2(pred_f, s=(h, w), dim=(-2, -1), norm="ortho").reshape(b, t, c, h, w)
    if global_pers_weight > 0:
        out = (1.0 - global_pers_weight) * out + global_pers_weight * ref
    if squeeze_t:
        out = out[:, 0]
    return out.to(dtype=pred_field.dtype)


def apply_inference_postprocess(
    pred_field: torch.Tensor,
    pred_traj: torch.Tensor,
    batch: dict[str, Any],
    cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    post_cfg = _parse_post_cfg(cfg)
    if not bool(post_cfg.get("enabled", False)):
        return pred_field, pred_traj

    field_out = pred_field
    traj_out = pred_traj
    if "traj_hist" in batch and isinstance(batch["traj_hist"], torch.Tensor):
        traj_out = _traj_postprocess(pred_traj=traj_out, traj_hist=batch["traj_hist"], post_cfg=post_cfg)
    if "x_hist" in batch and isinstance(batch["x_hist"], torch.Tensor):
        field_out = _field_spectral_postprocess(pred_field=field_out, x_hist=batch["x_hist"], post_cfg=post_cfg)
    return field_out, traj_out
