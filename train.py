from util import prepare_channel_tx_ris, find_indices
from util import discretize_phase, compute_sdma_performance
from core import RISnetPI, RTChannelsWMMSE
from core import RISnetPartialCSI4x4 as RISnetPartialCSI
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from params import params4users as params
import argparse
import datetime
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False
record = False and tb


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsnr")
    parser.add_argument("--ris_shape")
    parser.add_argument("--weights")
    parser.add_argument("--lr")
    parser.add_argument("--record")
    parser.add_argument("--device")
    parser.add_argument("--partialcsi")
    parser.add_argument("--trainingchannelpath")
    parser.add_argument("--testingchannelpath")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.tsnr is not None:
        params["tsnr"] = float(args.tsnr)
    if args.lr is not None:
        params["lr"] = float(args.lr)
    if args.weights is not None:
        weights = args.weights.split(',')
        params["alphas"] = np.array([float(w) for w in weights])
    if args.ris_shape is not None:
        ris_shape = args.ris_shape.split(',')
        params["ris_shape"] = tuple([int(s) for s in ris_shape])
    if args.record is not None:
        record = args.record == "True"
    else:
        record = False
    if args.partialcsi is not None:
        params["partial_csi"] = args.partialcsi == "True"
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.trainingchannelpath is not None:
        params['channel_ris_rx_path'] = args.trainingchannelpath
    if args.testingchannelpath is not None:
        params['channel_ris_rx_testing_path'] = args.testingchannelpath
    tb = record

    if record:
        now = datetime.datetime.now()
        if args.name is not None:
            dt_string = args.name
        else:
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        Path(params["results_path"] + dt_string).mkdir(parents=True, exist_ok=True)
        params["results_path"] = params["results_path"] + dt_string + "/"

    params["discrete_phases"] = params["discrete_phases"].to(device)

    if params["partial_csi"]:
        model = RISnetPartialCSI(params, device).to(device)
    else:
        model = RISnetPI(params, device).to(device)

    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)
    training_set = RTChannelsWMMSE(params, channel_tx_ris_pinv, device)
    test_set = RTChannelsWMMSE(params, channel_tx_ris_pinv, device, test=True)
    result_name = "ris_" + str(params['tsnr']) + "_" + str(params['ris_shape']) + '_' + str(params['alphas']) + "_"
    training_loader = DataLoader(dataset=training_set, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=256, shuffle=True)
    losses = list()
    if tb and record:
        writer = SummaryWriter(logdir=params["results_path"])
    counter = 1
    model.train()
    optimizer_wmmse = optim.Adam(model.parameters(), params['lr'])
    scheduler = ReduceLROnPlateau(optimizer_wmmse, 'min', min_lr=1e-8, patience=100, factor=0.5)

    if params["partial_csi"]:
        sensor_indices = find_indices(params["indices_sensors"], params["indices_sensors"], params["ris_shape"][1])
    else:
        sensor_indices = np.arange(params["ris_shape"][0] * params["ris_shape"][1])

    if params["mutual_coupling"]:
        sii = torch.load("data/mutual_coupling.pt", map_location=torch.device(device))
    else:
        sii = None

    # Training with WMMSE precoder
    training_underway = True
    while training_underway:
        training_set.wmmse_precode(model, channel_tx_ris, device, 5, sensor_indices)
        test_set.wmmse_precode(model, channel_tx_ris, device, 5, sensor_indices)
        for batch in training_loader:
            (sample_indices, channels_ris_rx_array, channels_ris_rx, channels_direct, location, weights,
             precoding) = batch
            optimizer_wmmse.zero_grad()
            nn_raw_output = model(channels_ris_rx_array[:, :, :, sensor_indices])
            wsr = compute_sdma_performance(nn_raw_output, channel_tx_ris, channels_ris_rx, channels_direct,
                                           precoding, sii, weights, params, device)
            loss = -torch.mean(wsr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any():
                    print("nan gradient found")
            optimizer_wmmse.step()
            scheduler.step(-wsr.mean())

            with torch.no_grad():
                for batch in test_loader:
                    (sample_indices, channels_ris_rx_array_test, channels_ris_rx_test, channels_direct_test,
                     location_test, weights_test, precoding_test) = batch
                    nn_raw_output_test = model(channels_ris_rx_array_test[:, :, :, sensor_indices])
                    wsr_test = compute_sdma_performance(nn_raw_output_test, channel_tx_ris, channels_ris_rx_test,
                                                        channels_direct_test, precoding_test, sii, weights_test,
                                                        params, device)

                    discrete_phases = discretize_phase(nn_raw_output_test, np.pi, device)
                    wsr_test_pi = compute_sdma_performance(discrete_phases, channel_tx_ris, channels_ris_rx_test,
                                                           channels_direct_test, precoding_test, sii, weights_test,
                                                           params, device)

                    discrete_phases = discretize_phase(nn_raw_output_test, np.pi / 2, device)
                    wsr_test_pid2 = compute_sdma_performance(discrete_phases, channel_tx_ris, channels_ris_rx_test,
                                                             channels_direct_test, precoding_test, sii, weights_test,
                                                             params, device)

            print('WMMSE round {round}, Epoch {epoch}, '
                  'data rate = {loss}'.format(round=counter,
                                              loss=-loss,
                                              epoch=counter))

            if tb and record:
                writer.add_scalar("Training/data_rate", -loss.item(), counter)
                writer.add_scalar("Training/learning_rate",
                                  optimizer_wmmse.param_groups[0]["lr"], counter)
                writer.add_scalar("Testing/data_rate_continuous", wsr_test.mean().item(), counter)
                writer.add_scalar("Testing/data_rate_pi", wsr_test_pi.mean().item(), counter)
                writer.add_scalar("Testing/data_rate_pid2", wsr_test_pid2.mean().item(), counter)

            counter += 1
            if counter > params["iter_wmmse"]:
                training_underway = False

            if record and counter % params['wmmse_saving_frequency'] == 0:
                torch.save(model.state_dict(), params['results_path'] + result_name +
                           'WMMSE_{iter}'.format(iter=counter))
                torch.save(training_set.v, params["results_path"] + result_name + "precoding_training.pt")
                torch.save(test_set.v, params["results_path"] + result_name + "precoding_testing.pt")
                if tb:
                    writer.flush()


if __name__ == "__main__":
    train()