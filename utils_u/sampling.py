
import torch


def generalized_steps(model, x, time, numstep=10,device='cuda'):
    with torch.no_grad():
        num_steps = numstep
        skip = time // num_steps
        epi = torch.randn_like(x)

        seq = range(0, time, skip)
        n = epi.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [epi]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            xt = xs[-1].to(device)

            x0_t = model(xt, x, t)
            xs.append(x0_t[:, 0, :])
        final_prototype = xs[-1]


    return final_prototype
