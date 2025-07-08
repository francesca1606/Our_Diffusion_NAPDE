import torch
import numpy as np
from scipy.optimize import curve_fit

# This code implements the correlation model described in the paper:
# Interfrequency Correlations among Fourier Spectral Ordinates and Implications for Stochastic Ground‐Motion Simulation
# Peter J. Stafford
# https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/107/6/2774/519322/Interfrequency-Correlations-among-Fourier-Spectral?redirectedFrom=fulltext


# --------------------------COEFFICIENTS-----------------------------------------

g0_E = -0.4690
g1_E = 0.1275
g2_E = 0.7793
g3_E = 3.0324
g4_E = 0.0106
s0_E = 0.8757
s1_E = -0.3309
s2_E = 0.5871
s3_E = 5.4264
s4_E = 0.5177
s5_E = 16.357
s6_E = 1.4689

n0_S = 0.1126
n1_S = -0.1068
g0_S = -0.6968
g1_S = -0.2963
g2_S = 0.3326
g3_S = 5.0078
g4_S = 0.5703
g5_S = 2.3973
g6_S = 3.5779
s0_S = 0.6167
s1_S = -0.1495
s2_S = 0.7248
s3_S = 3.6958
s4_S = 0.3640
s5_S = 13.457
s6_S = 2.2497

n0_A = 0.7606
n1_A = 0.2993
n2_A = 1.5837
n3_A = -0.5094
n4_A = 24.579
n5_A = 2.3518
g0_A = 0.9515
g1_A = 1.2752
g2_A = 1.4802
g3_A = -0.3361
s0_A = 0.7260
s1_A = 0.0328

# ------------------------Brune Model-------------------------------------------


def brune_model(f, M0, fc, tstar, C):
    return C * (M0 / (1 + (f / fc)**2)) * np.exp(-np.pi * f * tstar)


def Mw_to_M0(Mw):
    return 10**(Mw*1.5+9.1)


def fc_estimate(frequenze, dati_frequenza, magnitude):
    f = frequenze.numpy()
    S = torch.norm(dati_frequenza, dim=0).numpy()
    initial_M0 = Mw_to_M0(magnitude[index]).numpy()
    initial_fc = 5
    initial_tstar = 1
    initial_C = 0.1
    # Initial guess for parameters (you might have better initial estimates)
    initial_guess = [initial_M0, initial_fc, initial_tstar, initial_C]
    # Perform curve fitting to estimate parameters
    try:
        popt, pcov = curve_fit(brune_model, f, S, p0=initial_guess)
    except RuntimeError:
        print("Optimization failed. Try adjusting initial parameters or constraints.")
    estimated_M0, estimated_fc, estimated_tstar, estimated_C = popt
    return estimated_fc


# ------------------------Helper functions------------------------------

def S(f, a1, a2, a3):
    return a1/(1+torch.exp(-a3*torch.log(f/a2)))


def gamma_E(f):
    return g0_E + S(f, g1_E, g2_E, g3_E) + g4_E*torch.log(f/g2_E)


def corr_b_E(fi, fj, fc):
    f_min = torch.minimum(fi, fj)
    f_max = torch.maximum(fi, fj)
    f_min_ = f_min/fc
    f_max_ = f_max/fc
    gamma = gamma_E(f_min_)
    return torch.exp(gamma*torch.log(f_max_/f_min_))


def variance_b_E(f):
    return s0_E + S(f, s1_E, s2_E, s3_E) + S(f, s4_E, s5_E, s6_E)


def eta_S(f):
    return n0_S*torch.log(torch.maximum(torch.minimum(f, torch.tensor(4.)), torch.tensor(1/4))/0.25) + n1_S*torch.log(torch.maximum(f, torch.tensor(4.))/4)


def gamma_S(f):
    return g0_S + S(f, g1_S, g2_S, g3_S) + S(f, g4_S, g5_S, g6_S)


def rho0_S(fmax, fmin):
    eta = eta_S(fmin)
    gamma = gamma_S(fmin)
    return (1-eta)*torch.exp(gamma*torch.log(fmax/fmin))


def corr_b_S(fi, fj):
    f_min = torch.minimum(fi, fj)
    f_max = torch.maximum(fi, fj)
    return rho0_S(f_max, f_min) + (1 - rho0_S(f_min, f_min)) * torch.exp(-50*torch.log(f_max/f_min))


def variance_b_S(f):
    return s0_S + S(f, s1_S, s2_S, s3_S) + S(f, s4_S, s5_S, s6_S)


def eta_A(f):
    return S(f, n0_A, n1_A, n2_A) + (1 + S(f, n3_A, n4_A, n5_A))


def gamma_A(f):
    return S(torch.minimum(f, torch.tensor(10.)), g0_A, g1_A, g2_A) - 1 + g3_A*(torch.log(torch.maximum(f, torch.tensor(10.))/10))**2


def rho0_A(fmax, fmin):
    eta = eta_A(fmin)
    gamma = gamma_A(fmin)
    return (1-eta)*torch.exp(gamma*torch.log(fmax/fmin))


def corr_w_A(fi, fj):
    f_min = torch.minimum(fi, fj)
    f_max = torch.maximum(fi, fj)
    return rho0_A(f_max, f_min) + (1 - rho0_S(f_min, f_min)) * torch.exp(-50*torch.log(f_max/f_min))


def variance_w_A(f):
    return s0_A + s1_A*(torch.log(torch.maximum(f, torch.tensor(5.))/5))**2


def var_tot(f):
    return torch.sqrt(variance_b_E(f)**2 + variance_b_S(f)**2 + variance_w_A(f)**2)


def correlation(fi, fj, fc):
    return (corr_b_E(fi, fj, fc)*variance_b_E(fi)*variance_b_E(fj) + corr_b_S(fi, fj)*variance_b_S(fi)*variance_b_S(fj) + corr_w_A(fi, fj)*variance_w_A(fi)*variance_w_A(fj))/(var_tot(fi)*var_tot(fj))


def Loss_Covariant(v, alphas, sigmas, noised_reals, fc, device=None):

    v = v.to(device)

    # reconstruct from v the predicted clean image x0
    x0 = alphas * noised_reals - sigmas * v

    delta_f_T = 0.01   # frequency spacing
    T = x0.shape[-1]
    print(T)
    fs = T * delta_f_T  # sampling frequency

    dati_frequenza = torch.fft.fft(x0, dim=-1).real
    frequenze = torch.fft.fftfreq(T, d=1/fs).to(device)

    positivi = frequenze > 0
    frequenze = frequenze[positivi]

    n = len(frequenze)
    log_frequencies = torch.logspace(
        torch.log10(frequenze[1]).item(),
        torch.log10(frequenze[n // 2]).item(),
        steps=n // 2,
        base=10.0,
        device=device
    )

    indici_log_frequencies = torch.searchsorted(frequenze, log_frequencies)
    dati_corrispondenti = dati_frequenza[:, :, indici_log_frequencies]

    X = dati_corrispondenti.flatten(0, 1)
    cov_matrix = torch.cov(X.T)
    std_dev = torch.sqrt(torch.diag(cov_matrix))
    correlation_matrix = cov_matrix / torch.outer(std_dev, std_dev)

    correlation_empirical = correlation(
        log_frequencies.unsqueeze(1), log_frequencies.unsqueeze(0), fc)

    return torch.mean((correlation_matrix - correlation_empirical) ** 2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "data/6000_data/broadband.pt"
    data = torch.load(file_path, map_location=device)
    index = 0
    v = data[index, 0, :]
    fc = 1.
    loss = Loss_Covariant(v, fc)
    print(loss)
