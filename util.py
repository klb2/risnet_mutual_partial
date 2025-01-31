import numpy as np
import torch
from scipy.linalg import eigh
import copy


solver = True
try:
    from scipy.optimize import fsolve
except:
    solver = False


def mmse_precoding(channel, params, device='cpu'):
    if type(channel) is np.ndarray:
        channel = torch.from_numpy(channel).to(device)
    eye = torch.eye(channel.shape[1]).repeat((channel.shape[0], 1, 1)).to(device)
    p = channel.transpose(1, 2).conj() @ torch.linalg.inv(channel @ channel.transpose(1, 2).conj() +
                                                          1 / params['tsnr'] * eye)
    trace = torch.sum(torch.diagonal((p @ p.transpose(1, 2).conj()), dim1=1, dim2=2).real, dim=1, keepdim=True)
    p = p / torch.unsqueeze(torch.sqrt(trace), dim=2)
    return p


def cp2array_risnet(cp, factor=1, mean=0, device="cpu"):
    # Input: (batch, antenna)
    # Output: (batch, feature, antenna))
    array = torch.cat([(cp.abs() - mean) * factor, cp.angle() * 0.55], dim=1)

    return array.to(device)


def prepare_channel_direct_features(channel_direct, channel_tx_ris_pinv, params, device='cpu'):
    equivalent_los_channel = channel_direct @ channel_tx_ris_pinv
    return cp2array_risnet(equivalent_los_channel, 1 / params['std_direct'], params["mean_direct"], device)


def weighted_sum_rate(complete_channel, precoding, weights, params):
    channel_precoding = complete_channel @ precoding
    channel_precoding = torch.square(channel_precoding.abs())
    wsr = 0
    num_users = channel_precoding.shape[1]
    for user_idx in range(num_users):
        wsr += weights[:, user_idx] * torch.log2(1 + channel_precoding[:, user_idx, user_idx] /
                                                       (torch.sum(channel_precoding[:, user_idx, :], dim=1)
                                                        - channel_precoding[:, user_idx, user_idx]
                                                        + 1 / params["tsnr"]))
    return wsr


def compute_wmmse_v_v2(h_as_array, init_v, tx_power, noise_power, params, num_iters=500):
    num_users, num_tx_antennas = h_as_array.shape
    h_list = [h_as_array[user_idx: (user_idx + 1), :] for user_idx in range(num_users)]
    v_list = [init_v[:, user_idx: (user_idx + 1)] for user_idx in range(num_users)]
    w_list = [1 for _ in range(num_users)]
    for iter in range(num_iters):
        w_list_old = copy.deepcopy(w_list)

        # Step 2
        u_list = list()
        for user_idx in range(num_users):
            inv_hvvhi = (1 / (np.sum([np.real(h_list[user_idx] @ v
                                              @ v.transpose().conj() @ h_list[user_idx].transpose().conj())
                                      for v in v_list]) + noise_power))
            u_list.append(inv_hvvhi * h_list[user_idx] @ v_list[user_idx])

        # Step 3
        for user_idx in range(num_users):
            w_list[user_idx] = 1 / np.real(1 - u_list[user_idx].transpose().conj()
                                           @ h_list[user_idx] @ v_list[user_idx])

        # Step 4
        mmu = sum([alpha * h.transpose().conj() @ u @ w @ u.transpose().conj() @ h for alpha, h, u, w, in
                   zip(params["alphas"], h_list, u_list, w_list)])
        mphi = sum([alpha ** 2 * h.transpose().conj() @ u @ w ** 2 @ u.transpose().conj() @ h for alpha, h, u, w in
                    zip(params["alphas"], h_list, u_list, w_list)])

        try:
            lambbda, d = eigh(mmu)
        except:
            break
        lambbda = np.real(lambbda)
        phi = d.transpose().conj() @ mphi @ d
        phi = np.real(np.diag(phi))
        if solver:
            mu = fsolve(solve_mu, 0, args=(phi, lambbda, tx_power))
        else:
            raise ImportError('scipy.optimize.fsolve cannot be imported.')
        mv = np.linalg.inv(mmu + mu * np.eye(num_tx_antennas))

        v_list = [alpha * mv @ h.transpose().conj() @ u @ w for alpha, h, u, w in
                  zip(params["alphas"], h_list, u_list, w_list)]

        if np.sum([np.abs(w - w_old) for w, w_old in zip(w_list, w_list_old)]) < np.abs(w_list[0]) / 100 and iter > 500:
            break

    precoding = np.hstack(v_list)
    power = np.sum(np.abs(precoding) ** 2)
    return precoding / np.sqrt(power)


def solve_mu(mu, *args):
    phi = args[0]
    lambbda = args[1]
    p = args[2]
    return np.sum(phi / (lambbda + mu + 1e-3) ** 2) - p


def compute_complete_channel(channel_tx_ris, nn_output, channel_ris_rx, channel_direct):
    phi = torch.exp(1j * nn_output)
    complete_channel = (channel_ris_rx * phi) @ channel_tx_ris + channel_direct
    return complete_channel


def compute_complete_channel_mutual(channel_tx_ris, nn_output, channel_ris_rx, channel_direct, sii, params, device="cpu"):
    phi = torch.exp(1j * nn_output)
    identity = torch.eye(params["num_ris_antennas"]).to(device)
    to_be_inversed = identity - phi[:, :, None] * sii
    x = torch.linalg.solve(to_be_inversed[:, 0, :, :], channel_ris_rx, left=False)
    complete_channel = x * phi @ channel_tx_ris + channel_direct
    return complete_channel


def prepare_channel_tx_ris(params, device="cpu"):
    channel_tx_ris = torch.load(params['channel_tx_ris_path'], map_location=torch.device(device)).cfloat()
    channel_tx_ris = channel_tx_ris[:params["ris_shape"][0] * params["ris_shape"][1], :]
    channel_tx_ris_pinv = torch.linalg.pinv(channel_tx_ris)
    return channel_tx_ris, channel_tx_ris_pinv


def find_indices(x, y, n_cols):
    x, y = np.meshgrid(x, y)
    return (x + y * n_cols).flatten()


def discretize_phase(phases, granularity, device="cpu"):
    eligible_phases = torch.arange(0, 2 * torch.pi, granularity).to(device)
    eligible_vectors = torch.exp(1j * eligible_phases)
    eligible_vectors = eligible_vectors[None, :, None]
    phis = torch.exp(1j * phases)
    prod = eligible_vectors.real * phis.real + eligible_vectors.imag * phis.imag
    chosen_phases = torch.argmax(prod, dim=1, keepdim=True)
    chosen_phases = chosen_phases.float()
    chosen_phases *= granularity
    return chosen_phases


def compute_sdma_performance(nn_raw_output, channel_tx_ris, channels_ris_rx, channels_direct, precoding, sii, weights,
                             params, device="cpu"):
    if sii is None:
        complete_channel = compute_complete_channel(channel_tx_ris, nn_raw_output,
                                                    channels_ris_rx, channels_direct)
    else:
        complete_channel = compute_complete_channel_mutual(channel_tx_ris, nn_raw_output,
                                                           channels_ris_rx, channels_direct, sii, params, device)
    wsr = weighted_sum_rate(complete_channel, precoding, weights, params)
    return wsr
