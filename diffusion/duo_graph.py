import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


SAMPLING_RATE = 100


def amplitude_graph_STEAD(y, generate, path, idx, normalized=False):
    """
    Amplitude graph for STEAD dataset.

    This function generates amplitude graphs for the STEAD dataset, comparing the original signal `y` with the generated signal `generate`.
    It saves the graphs in the specified `path` with a unique identifier `idx`.

    Args:
        y (torch.Tensor): The original signal tensor of shape [num_channels, time_steps].
        generate (torch.Tensor): The generated signal tensor of shape [num_channels, time_steps].
        path (str): The directory path where the graphs will be saved.
        idx (int): An identifier for the saved graph files.
        normalized (bool, optional): If True, the y-axis will be labeled with "a(t)[1]", otherwise "a(t)[m/s²]". Defaults to False.
    """
    direction = ["E-W", "N-S", "U-D"]
    num_channels = y.shape[0]
    for i in range(num_channels):
        # plt.figure(figsize=(25, 10))
        plt.figure(figsize=(7, 4))
        ax = plt.subplot(1, 1, 1)

        loss = F.mse_loss(y[i], generate[i])
        ax.plot(generate[i].detach().cpu(),
                label=r'$\hat{y}$', color="blue", linewidth=2, zorder=1)
        ax.plot(y[i].detach().cpu(), label=r'$y$',
                color="red", linewidth=1, zorder=2)
        ax.set_title(direction[i], fontsize=10)
        ax.set_xlabel("t[s]", fontsize=15)
        if normalized:
            ax.set_ylabel(f"a(t)[1]", fontsize=15)
        else:
            ax.set_ylabel(f"a(t)[m/s²]", fontsize=15)
        ax.legend(fontsize=10, loc='lower left')
        ax.set_xticks(
            np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60])*100)
        # ax.set_yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
        ax.set_xticklabels([0, 4, 8, 12, 16, 20, 24, 28,
                           32, 36, 40, 44, 48, 52, 56, 60],)
        # ax.set_yticklabels(ax.get_yticks(), fontsize=15)

        plt.savefig(f"{path}/Amplitude_graph_{idx}_{direction[i]}.png")
        plt.close()


def frequency_loglog_STEAD(y, generate, path, idx, normalized=False):
    """
    Frequency log-log graph for STEAD dataset.

    This function generates frequency log-log graphs for the STEAD dataset, comparing the original signal `y` with the generated signal `generate`.
    It saves the graphs in the specified `path` with a unique identifier `idx`.

    Args:
        y (torch.Tensor): The original signal tensor of shape [num_channels, time_steps].
        generate (torch.Tensor): The generated signal tensor of shape [num_channels, time_steps].
        path (str): The directory path where the graphs will be saved.
        idx (int): An identifier for the saved graph files.
        normalized (bool, optional): If True, the y-axis will be labeled with "A(f)[1]", otherwise "A(f)[m/s²]". Defaults to False.
    """
    num_channels = y.shape[0]
    direction = ["E-W", "N-S", "U-D"]

    for i in range(num_channels):
        # plt.figure(figsize=(25, 10))
        plt.figure(figsize=(14, 8))
        axs = plt.subplot(1, 1, 1)
        # Extract the i-th channel
        cpu_y = y[i].cpu()
        cpu_generate = generate[i].cpu()

        # Conduct FFT
        y_fft_values = torch.abs(torch.fft.rfft(cpu_y) * 0.01)
        generate_fft_values = torch.abs(torch.fft.rfft(cpu_generate) * 0.01)

        # Calculate frequencies for FFT values
        freqs = torch.fft.rfftfreq(y_fft_values.numel(), d=1.0/SAMPLING_RATE)

        # Convert tensors to numpy arrays for plotting
        freqs_np = freqs.cpu().numpy()
        y_fft_np = y_fft_values.cpu().numpy()
        generate_fft_np = generate_fft_values.cpu().numpy()

        # Ensure that the shapes of freqs and y_fft_values match
        if freqs_np.shape[0] != y_fft_np.shape[0]:
            # Truncate or pad the arrays as necessary
            min_len = min(freqs_np.shape[0], y_fft_np.shape[0])
            freqs_np = freqs_np[:min_len]
            y_fft_np = y_fft_np[:min_len]
            generate_fft_np = generate_fft_np[:min_len]

        # Plot the data
        axs.loglog(freqs_np, generate_fft_np,
                   label=r'$\hat{y}$', color="blue", linewidth=2, zorder=1)
        axs.loglog(freqs_np, y_fft_np, label=r'$y$',
                   color="red", linewidth=1, zorder=2)
        # the title and labels
        axs.set_title(direction[i], fontsize=10)
        axs.set_xlabel('f[Hz]', fontsize=20)
        if normalized:
            axs.set_ylabel(f"A(f)[1]", fontsize=20)
        else:
            axs.set_ylabel(f"A(f)[m/s²]", fontsize=20)
        axs.set_xlim(0.1, 30)
        axs.set_ylim(10.**-3, 10**5)
        axs.legend(fontsize=15, loc='lower left')
        axs.set_yticks([10e-5, 10e-4, 10e-3, 10e-2, 10e-1,
                       10e-0, 10e1, 10e2, 10e3, 10e4, 10e5])
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(axs.get_xticks(), fontsize=15)
        axs.set_yticklabels(axs.get_yticks(), fontsize=15)
        # axs.set_yticklabels(["1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0,1,1e-2,1e1,1e2,1e3,"], fontsize=15)
        # Adjust layout
        plt.tight_layout()
        title = f'{path}/Frequency_Spectrumloglog{idx}_{direction[i]}.png'
        plt.savefig(title)
        plt.close()


def amplitude_graph_pbs(y, path, idx, normalized=False):
    """
    Amplitude graph for pbs dataset.

    This function generates amplitude graphs for the pbs dataset.
    It saves the graphs in the specified `path` with a unique identifier `idx`.

    Args:
        y (torch.Tensor): The original signal tensor of shape [num_channels, time_steps].
        path (str): The directory path where the graphs will be saved.
        idx (int): An identifier for the saved graph files.
        normalized (bool, optional): If True, the y-axis will be labeled with "a(t)[1]", otherwise "a(t)[m/s²]". Defaults to False.
    """
    direction = ["E-W", "N-S", "U-D"]
    num_channels = y.shape[0]
    for i in range(num_channels):
        # plt.figure(figsize=(25, 10))
        plt.figure(figsize=(7, 4))
        ax = plt.subplot(1, 1, 1)

        ax.plot(y[i].detach().cpu(), label=r'$\hat{y}$',
                color="blue", linewidth=1, zorder=2)
        ax.set_title(direction[i], fontsize=10)
        ax.set_xlabel("t[s]", fontsize=15)
        if normalized:
            ax.set_ylabel(f"a(t)[1]", fontsize=15)
        else:
            ax.set_ylabel(f"a(t)[m/s²]", fontsize=15)
        ax.legend(fontsize=10, loc='lower left')
        ax.set_xticks(
            np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60])*100)
        # ax.set_yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
        ax.set_xticklabels([0, 4, 8, 12, 16, 20, 24, 28,
                           32, 36, 40, 44, 48, 52, 56, 60],)
        # ax.set_yticklabels(ax.get_yticks(), fontsize=15)

        plt.savefig(f"{path}/Amplitude_graph_{idx}_{direction[i]}.png")
        plt.close()


def frequency_loglog_pbs(y, path, idx, normalized=False):
    """
    Frequency log-log graph for PBS dataset.

    This function generates frequency log-log graphs for the PBS dataset.
    It saves the graphs in the specified `path` with a unique identifier `idx`.

    Args:
        y (torch.Tensor): The original signal tensor of shape [num_channels, time_steps].
        path (str): The directory path where the graphs will be saved.
        idx (int): An identifier for the saved graph files.
        normalized (bool, optional): If True, the y-axis will be labeled with "A(f)[1]", otherwise "A(f)[m/s²]". Defaults to False.
    """
    num_channels = y.shape[0]
    direction = ["E-W", "N-S", "U-D"]

    for i in range(num_channels):
        # plt.figure(figsize=(25, 10))
        plt.figure(figsize=(14, 8))
        axs = plt.subplot(1, 1, 1)
        # Extract the i-th channel
        cpu_y = y[i].cpu()

        # Conduct FFT
        y_fft_values = torch.abs(torch.fft.rfft(cpu_y) * 0.01)

        # Calculate frequencies for FFT values
        freqs = torch.fft.rfftfreq(y_fft_values.numel(), d=1.0/SAMPLING_RATE)

        # Convert tensors to numpy arrays for plotting
        freqs_np = freqs.cpu().numpy()
        y_fft_np = y_fft_values.cpu().numpy()

        # Ensure that the shapes of freqs and y_fft_values match
        if freqs_np.shape[0] != y_fft_np.shape[0]:
            # Truncate or pad the arrays as necessary
            min_len = min(freqs_np.shape[0], y_fft_np.shape[0])
            freqs_np = freqs_np[:min_len]
            y_fft_np = y_fft_np[:min_len]

        # Plot the data
        axs.loglog(freqs_np, y_fft_np, label=r'$\hat{y}$',
                   color="blue", linewidth=1, zorder=2)
        # the title and labels
        axs.set_title(direction[i], fontsize=10)
        axs.set_xlabel('f[Hz]', fontsize=20)
        if normalized:
            axs.set_ylabel(f"A(f)[1]", fontsize=20)
        else:
            axs.set_ylabel(f"A(f)[m/s²]", fontsize=20)
        axs.set_xlim(0.1, 30)
        axs.set_ylim(10.**-3, 10**5)
        axs.legend(fontsize=15, loc='lower left')
        axs.set_yticks([10e-5, 10e-4, 10e-3, 10e-2, 10e-1,
                       10e-0, 10e1, 10e2, 10e3, 10e4, 10e5])
        axs.set_xticks([0.1, 1, 10])
        axs.set_xticklabels(axs.get_xticks(), fontsize=15)
        axs.set_yticklabels(axs.get_yticks(), fontsize=15)
        # axs.set_yticklabels(["1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,0,1,1e-2,1e1,1e2,1e3,"], fontsize=15)
        # Adjust layout
        plt.tight_layout()
        title = f'{path}/Frequency_Spectrumloglog{idx}_{direction[i]}.png'
        plt.savefig(title)
        plt.close()
