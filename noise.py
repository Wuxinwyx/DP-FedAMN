import torch
import math

def compute_updates_norm(model, global_params):
    updates = []
    sq, i = 0.0, 0
    with torch.no_grad():
        for p in model.module.parameters():                 
            if p.requires_grad:
                u = p.detach() - global_params[i]
                updates.append(u)
                sq += u.pow(2).sum().item()
                i += 1
        norm = (sq ** 0.5)
        return updates, norm

def clip_updates(updates, norm, dp_clip):
    scale = min(1.0, dp_clip / norm)
    if scale == 1.0:
        clipped = [u.clone() for u in updates]
    else:
        clipped = [u * scale for u in updates]
    return clipped

def compute_dp_sigma(args):
    dp_delta = 1 / args.n_parties
    # dp_delta = 0.01
    if args.dp_epsilon <= 0:
        raise ValueError("dp_epsilon must be > 0.")
    if not (0.0 < dp_delta < 1.0):
        raise ValueError("dp_delta must be in (0, 1).")

    sigma = (7.0 * (args.sample_fraction ** 2) * args.comm_round 
                * (args.dp_epsilon + 2.0 * math.log(1.0 / dp_delta)) / (args.dp_epsilon ** 2))
    return math.sqrt(sigma)

def add_noise(model, noise_list):
    i = 0
    with torch.no_grad():
        for p in model.module.parameters():
            if p.requires_grad:
                p.data.add_(noise_list[i])
            i += 1

def sub_noise(model, noise_list):
    i = 0
    with torch.no_grad():
        for p in model.module.parameters():
            if p.requires_grad:
                p.data.sub_(noise_list[i])
            i += 1