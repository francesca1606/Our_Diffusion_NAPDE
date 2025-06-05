import matplotlib.pyplot as plt
import numpy as np
import torch
from math import log10
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

SAMPLING_RATE = 100

def visualize_signal(signal):
    """
        Visualize the 3 axis of a signal.
    """
    x = signal[0]
    y = signal[1]
    z = signal[2]

    fig,ax = plt.subplots(3,1, figsize = (20,20))
    ax[0].plot(x)
    ax[1].plot(y)
    ax[2].plot(z)
    plt.show()


def save_frequency_graph(tensor_signal,path,idx, generated = False, k = False):
    num_channels = tensor_signal.shape[0]
    
    # Create the plot
    fig, axs = plt.subplots(num_channels, figsize=(10, 4*num_channels))

    for i in range(num_channels):
        # Extract the i-th channel
        channel_data = tensor_signal[i, :]

        # Conduct FFT
        fft_values = torch.fft.rfft(channel_data) * 0.01

        # Calculate absolute value of FFT values (for magnitude) and apply log scale
        magnitude = torch.abs(fft_values)  # added small value to prevent log(0)

        # Calculate frequencies for FFT values
        freqs = torch.fft.rfftfreq(channel_data.numel(), d=1.0/SAMPLING_RATE)
        #freqs = torch.log10(freqs)
        # Convert tensors to numpy arrays for plotting
        freqs = freqs.cpu().numpy()
        magnitude = magnitude.cpu().numpy()

        # Plot the data
        axs[i].plot(freqs, magnitude)

        # Set the title and labels
        axs[i].set_title(f'Log Frequency Spectrum - Channel {i+1}')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Log Magnitude')

    # Adjust layout
    plt.tight_layout()
    title = f'{path}/Generated_signal_{idx}_Frequency_Spectrum.png' if generated else f'{path}/Signal_{idx}_Frequency_Spectrum.png'
    plt.savefig(title)
    plt.close()
  

def save_amplitude_graph(tensor_signal,path,idx, generated = False):

    fig,ax = plt.subplots(3,1,figsize=(20,20))
    tensor_cpu = tensor_signal.cpu()
    ax[0].plot(tensor_cpu[0])
    ax[1].plot(tensor_cpu[1])
    ax[2].plot(tensor_cpu[2])
    title = f"{path}/Generated_signal_{idx}_Amplitude_Graph.png" if generated else f"{path}/Signal_{idx}_Amplitude_Graph.png"
    fig.savefig(title)
    plt.close()
    

def save_phase(tensor_signal,idx ,save = True, path=None,generated = False):
    # Apply Fast Fourier Transform
    
    # Extract phase
    # Plot
    fig, axs = plt.subplots(3, figsize=(14, 7))

    for i in range(tensor_signal.shape[0]):
        channel_data = tensor_signal[i, :]
        fft_result = torch.fft.fft(channel_data)
        phase = torch.angle(fft_result)

        
        freq =  torch.fft.fftfreq(channel_data.numel(), d = 1/ SAMPLING_RATE)
        
        freq = freq.cpu().numpy()
        phase = phase.cpu().numpy()
        axs[i].plot(freq,phase)

        axs[i].set_title(f'Phase of Channel {i+1}')

    plt.tight_layout()

    # Save plot to file
    if save and path is not None:
        title = f"{path}/Generated_signal_{idx}_phase_graph.png" if generated else f"{path}/Signal_{idx}_phase_graph.png"
        plt.savefig(title)

    #plt.show()

def save_frequency_graph_loglog(tensor_signal,path,idx, generated = False, k = False ):
    num_channels = tensor_signal.shape[0]
    
    # Create the plot
    fig, axs = plt.subplots(num_channels, figsize=(10, 4*num_channels))

    for i in range(num_channels):
        # Extract the i-th channel
        channel_data = tensor_signal[i, :]

        # Conduct FFT
        fft_values = torch.fft.rfft(channel_data) * 0.01

        # Calculate absolute value of FFT values (for magnitude) and apply log scale
        abs_fft_values = torch.abs(fft_values)  # added small value to prevent log(0)

        # Calculate frequencies for FFT values
        freqs = torch.fft.rfftfreq(channel_data.numel(), d=1.0/SAMPLING_RATE)
        #freqs = torch.log10(freqs)
        # Convert tensors to numpy arrays for plotting
        if k :
            from random import random
            a = 1e-6
            b = 2e-4
            n1 = (a + (b - a) * torch.rand(1)).item()
            n2 = (a + (b - a) * torch.rand(1)).item()
            abs_fft_values[300:] = abs_fft_values[300:] + (torch.rand_like(abs_fft_values[300:]) *n1 + n2)
        freqs = freqs.cpu().numpy()
        abs_fft_values = abs_fft_values.cpu().numpy()

        # Plot the data
        axs[i].loglog(freqs, abs_fft_values)
        axs[i].set_xlim()

        # Set the title and labels
        axs[i].set_title(f'Log Frequency Spectrum - Channel {i+1}')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Log Magnitude')

        axs[i].set_xlim(0.1,30) 
        axs[i].set_xticks(np.array([0.1,1.0,10.,50.]))
        axs[i].set_ylim(10.**-6,10.**1)
        axs[i].set_yticks(10.**np.arange(-6,1))

    # Adjust layout
    plt.tight_layout()
    title = f'{path}/Generated_signal_{idx}_Frequency_Spectrumloglog.png' if generated else f'{path}/Signal_{idx}_Frequency_Spectrumloglog.png'
    plt.savefig(title)
    plt.close()


def visualise_fft(tensor_signal ):
    num_channels = tensor_signal.shape[0]
    
    # Create the plot
    fig, axs = plt.subplots(num_channels, figsize=(10, 4*num_channels))

    for i in range(num_channels):
        # Extract the i-th channel
        channel_data = tensor_signal[i, :]

        # Conduct FFT
        fft_values = torch.fft.rfft(channel_data) * 0.01

        # Calculate absolute value of FFT values (for magnitude) and apply log scale
        #abs_fft_values = torch.abs(fft_values)  # added small value to prevent log(0)

        # Calculate frequencies for FFT values
        freqs = torch.fft.rfftfreq( channel_data.numel() ,d=1.0/SAMPLING_RATE)

        # Convert tensors to numpy arrays for plotting
        freqs = freqs.numpy()
        fft_values = fft_values.numpy()
        abs_fft_values = abs_fft_values.numpy()


        # Plot the data
        axs[i].plot(freqs, fft_values)

        # Set the title and labels
        axs[i].set_title(f'Frequency Spectrum - Channel {i+1}')
        axs[i].set_xlabel('Frequency (Hz)')
        axs[i].set_ylabel('Magnitude')

    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.show()


def visualisation_3d(tensor_signal, save = False, idx = None, path = ""):
    tensor_signal = tensor_signal.cpu()
    X = np.arange(tensor_signal.shape[1])
    Y = np.arange(tensor_signal.shape[0])
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Since we're working with 3D data, let's loop over each row (sub-signal) individually
    for i in range(tensor_signal.shape[0]):
        ax.plot3D(X[i], Y[i], tensor_signal[i], 'gray')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    if save :
        plt.savefig(f"{path}/3d_signal_{idx}")

    else : 
        plt.show()

    plt.close()





def visualize_low_high(low,high):

    fig,ax = plt.subplots(3,2, figsize = (20,20))

    for i in range(low.shape[0]):
        ax[i,0].plot(low[i])
        ax[i,1].plot(high[i])

    plt.show()
