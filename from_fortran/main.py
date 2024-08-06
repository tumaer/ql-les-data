"""Main function."""

import os

import numpy as np
from omegaconf import OmegaConf

from dimensions import nx, ny, nz, nx2, ny2, nz2, npmax, npx, npy, npz, npx2, npy2, npz2, ncpu


# from datafilenames import *
control_dfn = 'control'
parameter_dfn = 'param'
input_grid_dfn = 'grid_in.dat'
output_grid_dfn = 'grid_out.dat'
input_field_dfn = 'field_in.dat'
output_field_dfn = 'field_out.dat'
global_data_dfn = 'global_data.dat'
filtered_u_spectra_dfn = 'fil_u_spec.dat'
unfiltered_u_spectra_dfn = 'unfil_u_spec.dat'
filtered_v_spectra_dfn = 'fil_v_spec.dat'
unfiltered_v_spectra_dfn = 'unfil_v_spec.dat'
filtered_w_spectra_dfn = 'fil_w_spec.dat'
unfiltered_w_spectra_dfn = 'unfil_w_spec.dat'
filtered_e_spectra_dfn = 'fil_e_spec.dat'
unfiltered_e_spectra_dfn = 'unfil_e_spec.dat'


# inout.py
def LESDAT(cfg_root: str):
    control_dfn = os.path.join(cfg_root, "control.yaml")
    parameter_dfn = os.path.join(cfg_root, "param.yaml")

    cfg = OmegaConf.merge(OmegaConf.load(control_dfn), OmegaConf.load(parameter_dfn))

    global job_id_ctr, taumax, itmax, ioutst, ianalout, irunout, dtauout, tauout_0, tauout_1, cfl, tstufe_rq, itstep_rq, cyber
    global job_id_par, reynolds, faku, initial_data, ntopleft, ntopright, captau, itsec, isgs_model, alpha_prim, alpha_sec, beta_prim, beta_sec, itype_prim, itype_sec, iord_decon, iord_sec, iord_primreg

    job_id_ctr = cfg.job_id_ctr
    taumax = cfg.taumax
    itmax = cfg.itmax
    ioutst = cfg.ioutst
    ianalout = cfg.ianalout
    irunout = cfg.irunout
    dtauout = cfg.dtauout
    tauout_0 = cfg.tauout_0
    tauout_1 = cfg.tauout_1
    cfl = cfg.cfl
    tstufe_rq = cfg.tstufe_rq
    itstep_rq = cfg.itstep_rq
    cyber = cfg.cyber

    job_id_par = cfg.job_id_par
    reynolds = cfg.reynolds
    faku = cfg.faku
    initial_data = cfg.initial_data
    ntopleft = cfg.ntopleft
    ntopright = cfg.ntopright
    captau = cfg.captau
    itsec = cfg.itsec
    isgs_model = cfg.isgs_model
    alpha_prim = cfg.alpha_prim
    alpha_sec = cfg.alpha_sec
    beta_prim = cfg.beta_prim
    beta_sec = cfg.beta_sec
    itype_prim = cfg.itype_prim
    itype_sec = cfg.itype_sec
    iord_decon = cfg.iord_decon
    iord_sec = cfg.iord_sec
    iord_primreg = cfg.iord_primreg

    # print values
    print(OmegaConf.to_yaml(cfg))

    return cfg


def SETUP(cfg):

    global ifaxx, ifaxx_psd, ifaxx_n2, trigsx, trigsx_psd, trigsx_n2
    global ifaxy, ifaxy_psd, ifaxy_n2, trigsy, trigsy_psd, trigsy_n2
    global ifaxz, ifaxz_psd, ifaxz_n2, trigsz, trigsz_psd, trigsz_n2
    global ghat, ghat2, qhat
    global xrechen, xmet1, xmet2, yrechen, ymet1, ymet2, zrechen, zmet1, zmet2
    global hx, hy, hz
    global epsmax, cyber
    global alpha_prim, alpha_sec, beta_prim, beta_sec, itype_prim, itype_sec
    global iord_decon, iord_sec, iord_primreg
    global captau, itsec

    ifaxx = np.zeros(14, dtype=int)
    ifaxx_psd = np.zeros(14, dtype=int)
    ifaxx_n2 = np.zeros(14, dtype=int)
    trigsx = np.zeros(2 * nx + 1, dtype=float)
    trigsx_psd = np.zeros(3 * nx + 1, dtype=float)
    trigsx_n2 = np.zeros(2 * nx2 + 1, dtype=float)

    ifaxy = np.zeros(14, dtype=int)
    ifaxy_psd = np.zeros(14, dtype=int)
    ifaxy_n2 = np.zeros(14, dtype=int)
    trigsy = np.zeros(2 * ny + 1, dtype=float)
    trigsy_psd = np.zeros(3 * ny + 1, dtype=float)
    trigsy_n2 = np.zeros(2 * ny2 + 1, dtype=float)

    ifaxz = np.zeros(14, dtype=int)
    ifaxz_psd = np.zeros(14, dtype=int)
    ifaxz_n2 = np.zeros(14, dtype=int)
    trigsz = np.zeros(2 * nz + 1, dtype=float)
    trigsz_psd = np.zeros(3 * nz + 1, dtype=float)
    trigsz_n2 = np.zeros(2 * nz2 + 1, dtype=float)

    ghat = np.zeros((npx, npy, npz), dtype=complex)
    ghat2 = np.zeros((npx, npy, npz), dtype=complex)
    qhat = np.zeros((npx, npy, npz), dtype=complex)

    # find machine precision
    epsmax = 1.0
    while 1 < 1 + epsmax:
        epsmax *= 0.5
    epsmax *= 2.0
    print('machine precision is:', epsmax)

    # prepare FFTs -> not needed in Python, TODO: find out what this does.
    # call FFTFAX (nx, ifaxx, trigsx)
    # call CFTFAX (ny, ifaxy, trigsy)
    # call CFTFAX (nz, ifaxz, trigsz)

    # call FFTFAX (3*nx/2, ifaxx_psd, trigsx_psd)
    # call CFTFAX (3*ny/2, ifaxy_psd, trigsy_psd)
    # call CFTFAX (3*nz/2, ifaxz_psd, trigsz_psd)

    # call FFTFAX (nx2, ifaxx_n2, trigsx_n2)
    # call CFTFAX (ny2, ifaxy_n2, trigsy_n2)
    # call CFTFAX (nz2, ifaxz_n2, trigsz_n2)

    # assemble mesh
    pi2 = 2 * np.pi
    xrechen = np.zeros(npx, dtype=float)
    xmet1 = np.zeros(npx, dtype=float)
    xmet2 = np.zeros(npx, dtype=float)
    yrechen = np.zeros(npy, dtype=float)
    ymet1 = np.zeros(npy, dtype=float)
    ymet2 = np.zeros(npy, dtype=float)
    zrechen = np.zeros(npz, dtype=float)
    zmet1 = np.zeros(npz, dtype=float)
    zmet2 = np.zeros(npz, dtype=float)

    for i in range(nx + 1):
        xrechen[i] = i * pi2 / nx
    for i in range(ny + 1):
        yrechen[i] = i * pi2 / ny
    for i in range(nz + 1):
        zrechen[i] = i * pi2 / nz

    hx = pi2 / nx
    hy = pi2 / ny
    hz = pi2 / nz

    # initial time
    global tauout
    tauout = 0.0

    # set up filter for LES or DNS (data processing)
    if cyber[3]:
        raise NotImplementedError
    else:
        ghat_p, ghat, ghat2 = UNINITFILTER()
        qhat = UNINITDECON()


def UNINITFILTER():
    global ghat_p, ghat, ghat2
    global alpha_prim, alpha_sec, beta_prim, beta_sec, itype_prim, itype_sec
    global iord_decon, iord_sec, iord_primreg
    global nx, ny, nz, npx, npy, npz
    global beta2

    beta2 = 1.0

    ghat_p = np.zeros((npx, npy, npz), dtype=float)

    pi = np.pi

    # construct postprocessing primary filter
    alpha = alpha_prim
    beta = beta_prim
    if itype_prim == 1:
        if beta > 0.0:
            alpha_prim = -0.5 * np.cos(pi / beta)
            alpha = alpha_prim
        if alpha != 0.0:
            beta_prim = pi / np.acos(-2.0 * alpha)
            beta = beta_prim
        print(' initializing primary filter for postproc: ')
        print(' Pade-type, with alpha =', alpha, ' beta =', beta)
        for k3 in range(-nz // 2, nz // 2):
            k = (k3 + nz) % nz + 1
            for k2 in range(-ny // 2, ny // 2):
                j = (k2 + ny) % ny + 1
                for k1 in range(0, nx // 2):
                    i = k1 * 2 + 1
                    g = 0.5 * (1.0 + 2.0 * alpha) * (np.cos(abs(k1) / (nx // 2) * pi) + 1.0) / (
                            1.0 + 2.0 * alpha *np.cos(abs(k1) / (nx // 2) * pi))
                    g = g * 0.5 * (1.0 + 2.0 * alpha) * (np.cos(abs(k2) / (ny // 2) * pi) + 1.0) / (
                            1.0 + 2.0 * alpha *np.cos(abs(k2) / (ny // 2) * pi))
                    g = g * 0.5 * (1.0 + 2.0 * alpha) * (np.cos(abs(k3) / (nz // 2) * pi) + 1.0) / (
                            1.0 + 2.0 * alpha *np.cos(abs(k3) / (nz // 2) * pi))
                    ghat_p[i-1, j-1, k-1] = g  # TODO: is this "-1" correct?
                    ghat_p[i + 1-1, j-1, k-1] = g
    elif itype_prim == 2:
        # print(' initializing primary filter for postproc: ')
        # print(' Gauss-type, with beta =', beta)
        # alphax = -24.0 * np.log(0.5) / (nx ** 2) * beta ** 2
        # alphay = -24.0 * np.log(0.5) / (ny ** 2) * beta ** 2
        # alphaz = -24.0 * np.log(0.5) / (nz ** 2) * beta ** 2
        # for k3 in range(-nz // 2, nz // 2):
        #     k = (k3 + nz) % nz + 1
        #     for k2 in range(-ny // 2, ny // 2):
        #         j = (k2 + ny) % ny + 1
        #         for k1 in range(0, nx // 2):
        #             i = k1 * 2 + 1
        #             if (k1 * beta2) <= (nx // 2) - 1.0 and (
        #                 k2 * beta2) <= (ny // 2) - 1.0 and (
        #                 k3 * beta2) <= (nz // 2) - 1.0 and (
        #                 k1 * beta2) >= -(nx // 2) and (
        #                 k2 * beta2) >= -(ny // 2) and (
        #                 k3 * beta2) >= -(nz // 2
        #             ):
        #                 g = np.exp(-1.0 / 24.0 * (
        #                     alphax * k1 ** 2 + alphay * k2 ** 2 + alphaz * k3 ** 2
        #                 ))
        #             else:
        #                 g = 0.0
        #             ghat_p[i-1, j-1, k-1] = g
        #             ghat_p[i + 1-1, j-1, k-1] = g
        raise NotImplementedError

    # neutralize active primary and secondary filters
    ghat = np.ones((npx, npy, npz), dtype=float)
    ghat2 = np.ones((npx, npy, npz), dtype=float)

    return ghat_p, ghat, ghat2


def UNINITDECON():
    global qhat
    qhat = np.ones((npx, npy, npz), dtype=float)
    return qhat


def incbox(cfg_root: str, data_root: str):

    # implicit   integer (i-n)
    # implicit   real (a-h,o-z)

    u = np.zeros((npx, npy, npz, 3), dtype=float)
    cyber_temp = np.zeros(25, dtype=bool)

    # re-allocate output data files if new run
    if not cyber_temp[0]:
        with open(output_grid_dfn, 'wb') as f:
            f.close()
        with open(output_field_dfn, 'wb') as f:
            f.close()
        with open(global_data_dfn, 'w') as f:
            f.close()
        with open(filtered_u_spectra_dfn, 'w') as f:
            f.close()
        with open(unfiltered_u_spectra_dfn, 'w') as f:
            f.close()
        with open(filtered_v_spectra_dfn, 'w') as f:
            f.close()
        with open(unfiltered_v_spectra_dfn, 'w') as f:
            f.close()
        with open(filtered_w_spectra_dfn, 'w') as f:
            f.close()
        with open(unfiltered_w_spectra_dfn, 'w') as f:
            f.close()
        with open(filtered_e_spectra_dfn, 'w') as f:
            f.close()
        with open(unfiltered_e_spectra_dfn, 'w') as f:
            f.close()

    # read parameters and setup problem
    cfg = LESDAT(cfg_root)  # TODO: pass as argument
    SETUP(cfg)

    # open concurrent graphics
    # if cfg.cyber[24]:
    #     istat = PGOPEN('/xw')

    # read re-start data or generate initial data
    if cfg.cyber[0]:  # read re-start data
        # EINGABE()
        # time = tau0
        raise NotImplementedError
    else:  # generate initial data
        u = ANFBED(u)
        global time, itstep
        time = 0.
        itstep = 0

        global tzwstf, deltat
        tzwstf = 0.
        deltat = 0.

    print(' ')
    print(' job_id = ', cfg.job_id_par)
    print(' reynolds = ', reynolds)

    # check initial data
    if cfg.cyber[3]:  # LES
        # print(' Analyse vor Filter')
        # print('ANALYSE at t = ', time)
        # print('      itstep = ', itstep)
        # ANALYSE(0, 1)
        # for l in range(3):
        #     FILTER(u[0, 0, 0, l])
        # print(' Analyse nach Filter')
        # print('ANALYSE at t = ', time)
        # print('      itstep = ', itstep)
        # ANALYSE(0, 1)
        raise NotImplementedError
    else:  # DNS
        print('ANALYSE at t = ', time)
        print('      itstep = ', itstep)
        # ANALYSE(0, 1)  # TODO: 

    # for LES make sure to use Helmholtz projection
    if cfg.cyber[3]:
        cfg.cyber[4] = True

    # time marching
    TINTEG(u)

    # end
    print("Simulation done!")

    return 0

def FFT991(u, work, trigsx, ifaxx, sign, npx, nx, npy_npz):
    # Multiple real/half-complex periodic fast Fourier transform.
    pass


def CFFT99(A,WORK,TRIGS,IFAX,INC,JUMP,N,LOT,ISIGN):
    # u, work, trigsy, ifaxy, npx_2, ny, npx_2_npy, sign
    # CFFT99 (A,WORK,TRIGS,IFAX,INC,JUMP,N,LOT,ISIGN
    # Perform multiple fast Fourier transforms.
    """
    Perform a complex Fast Fourier Transform (CFFT99).

    Args:
        A (ndarray): A complex array of length N*INC+(LOT-1)*JUMP containing the input 
            gridpoint or coefficient vectors. This array will be overwritten by the results.
        WORK (ndarray): A complex work array of length N*LOT or a real array of length 2*N*LOT. - remove
        TRIGS (ndarray): An array set up by CFTFAX, which must be called first.         - remove
        IFAX (ndarray): An array set up by CFTFAX, which must be called first.          - remove
        INC (int): The increment (in word pairs) between successive elements of each    - remove
            (complex) gridpoint or coefficient vector. For example, INC=1 for consecutively stored data.
        JUMP (int): The increment (in word pairs) between the first elements of         - remove
            successive data or coefficient vectors. On the Cray, try to arrange data so
            that JUMP is not a multiple of 8 (to avoid memory bank conflicts).
        N (int): The length of each transform.                                          - remove
        LOT (int): The number of transforms to be done simultaneously.                  - remove
        ISIGN (int): -1 for a transform from gridpoint values to Fourier coefficients,  - remove
            +1 for a transform from Fourier coefficients to gridpoint values.
    """

    pass


def energy_spectrum_imposed(E, nmax, initial_data, ntopleft, ntopright):
    # Calculate the energy spectrum with imposed initial data.

    # initial data for CBC
    pi = np.pi
    l = 55.0
    u_0 = 27.1893

    const1 = 2.0 * pi / l
    const2 = const1 / (u_0**2)

    a42 = [0.56102E+01, -0.11236E+01, -0.30961E+00, 0.33172E+00, -0.10959E+00, -0.22320E-01, 0.66575E-02]
    a98 = [0.43649E+01, -0.11793E+01, 0.67320E-01, 0.88554E-01, -0.13372E+00, 0.28385E-01, -0.99708E-02]
    a171 = [0.36567E+01, -0.11641E+01, -0.51571E-02, 0.38064E-02, -0.10484E+00, 0.12676E-01, -0.54562E-02]

    # initialize E(k)
    E = np.zeros(nmax + 1)  # 27+1

    # count modes on shells
    for k3 in range(-nz//2, nz//2):
        for k2 in range(-ny//2, ny//2):
            for k1 in range(-nx//2, nx//2):
                nk = round(np.sqrt(k1**2 + k2**2 + k3**2))  # up to 28
                E[nk] += 1.

    E[0] = 0.
    if initial_data == 0:  # white noise
        for k in range(nmax+1):
            E[k] = k**2 / E[k]

    elif initial_data == 1:  # Kolmogorov
        for k in range(1, nmax+1):
            E[k] = k**(-5/3) / E[k]  # (2 * np.pi * k**2)

    elif initial_data == 2:  # CBC 1
        for k in range(1, nmax+1):
            sum = 0.0
            for n in range(7):
                sum += a42[n] * np.log(const1 * k)**n
            E[k] = const2 * np.exp(sum) / E[k]

    elif initial_data == 3:  # CBC 2
        for k in range(nmax+1):
            sum = 0.0
            for n in range(7):
                sum += a98[n] * np.log(const1 * k)**n
            E[k] = const2 * np.exp(sum) / E[k]

    elif initial_data == 4:  # CBC 3
        for k in range(nmax+1):
            sum = 0.0
            for n in range(7):
                sum += a171[n] * np.log(const1 * k)**n
            E[k] = const2 * np.exp(sum) / E[k]

    elif initial_data == 6:  # Pao
        for k in range(nmax+1):
            kp = ntopleft
            E[k] = k**4 * np.exp(-2. * (k / kp)**2) / E[k]

    else:
        raise ValueError('ERROR bad initial_data in energy_spectrum_imposed')

    return E


def concurrent_plot(a, b):
    # Implementation of concurrent_plot function
    pass


def get_real_wavenumber_grid(N):
    Nf = N//2 + 1
    k = np.fft.fftfreq(N, 1./N)  # for y and z direction
    kx = k[:Nf].copy()
    kx[-1] *= -1
    k_field = np.array(np.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    return k_field, k

def get_energy_spectrum(vel):
    Nx, Ny, Nz = vel.shape[1:]
    assert (Nx == Ny and Ny == Nz)

    N = Nx
    Nf = N//2 + 1

    k_field, k = get_real_wavenumber_grid(N=Nx)
    k_mag = np.sqrt(k_field[0]*k_field[0] + k_field[1]
                    * k_field[1] + k_field[2]*k_field[2])

    shell = (k_mag + 0.5).astype(int)
    fact = 2 * (k_field[0] > 0) * (k_field[0] < N//2) + \
        1 * (k_field[0] == 0) + 1 * (k_field[0] == N//2)

    # Fourier transform
    vel_hat = np.stack([np.fft.rfftn(vel[ii], axes=(2, 1, 0))
                        for ii in range(3)])

    ek = np.zeros(N)
    n_samples = np.zeros(N)

    uu = fact * 0.5 * (np.abs(vel_hat[0]*vel_hat[0]) + np.abs(
        vel_hat[1]*vel_hat[1]) + np.abs(vel_hat[2]*vel_hat[2]))

    np.add.at(ek, shell.flatten(), uu.flatten())
    np.add.at(n_samples, shell.flatten(), 1)
    ek *= 4 * np.pi * k**2 / (n_samples + 1e-10)
    ek *= 1/(N**3)

    return ek

def ANFBED(u):
    global nx, ny, nz, npmax
    global ifaxx, ifaxx_psd, trigsx, trigsx_psd
    global ifaxy, ifaxy_psd, trigsy, trigsy_psd
    global ifaxz, ifaxz_psd, trigsz, trigsz_psd

    global cyber, faku, ntopleft, ntopright, initial_data

    work = np.zeros((nx, ny, nz))
    v = np.zeros((npx, npy, npz, 3))
    energ = np.zeros(npmax)
    energ_s = np.zeros(npmax)
    E = np.zeros(3*npmax)

    np.random.seed(760013)  # TODO: iranseed = 760013

    # Taylor-Green
    if initial_data == 5:
        # print('TG in xy')
        # # print('TG in xz')
        # # print('TG in yz')
        # # print('TG in xyz')
        # pi = 4. * np.arctan(1.)
        # facfft = 1. / (ny * nz)
        # for k in range(nz):
        #     for j in range(ny):
        #         for i in range(nx):
        #             u[i, j, k, 0] = -1. * np.cos(2. * pi * float(i) / float(nx)) * np.sin(2. * pi * float(j) / float(ny))
        #             u[i, j, k, 1] = 1. * np.sin(2. * pi * float(i) / float(nx)) * np.cos(2. * pi * float(j) / float(ny))
        #             u[i, j, k, 2] = 0.
        #             # u[i, j, k, 0] = -1. * np.cos(2. * pi * float(i) / float(nx)) * np.sin(2. * pi * float(k) / float(nz))
        #             # u[i, j, k, 2] = 1. * np.sin(2. * pi * float(i) / float(nx)) * np.cos(2. * pi * float(k) / float(nz))
        #             # u[i, j, k, 1] = 0.
        #             # u[i, j, k, 1] = -1. * np.cos(2. * pi * float(j) / float(ny)) * np.sin(2. * pi * float(k) / float(nz))
        #             # u[i, j, k, 2] = 1. * np.sin(2. * pi * float(j) / float(ny)) * np.cos(2. * pi * float(k) / float(nz))
        #             # u[i, j, k, 0] = 0.
        #             # u[i, j, k, 2] = 0.
        #             # u[i, j, k, 0] = np.sin(2. * pi * float(i) / float(nx)) * np.cos(2. * pi * float(j) / float(ny)) * np.cos(2. * pi * float(k) / float(nz))
        #             # u[i, j, k, 1] = -np.cos(2. * pi * float(i) / float(nx)) * np.sin(2. * pi * float(j) / float(ny)) * np.cos(2. * pi * float(k) / float(nz))
        #             # u[i, j, k, 0] = 0.
        #             # u[i, j, k, 1] = np.cos(2. * pi * float(i) / float(nx)) * np.sin(2. * pi * float(j) / float(ny)) * np.cos(2. * pi * float(k) / float(nz))
        #             # u[i, j, k, 2] = -np.cos(2. * pi * float(i) / float(nx)) * np.cos(2. * pi * float(j) / float(ny)) * np.sin(2. * pi * float(k) / float(nz))
        #             u[i, j, k, 0] = faku * u[i, j, k, 0]
        #             u[i, j, k, 1] = faku * u[i, j, k, 1]
        #             u[i, j, k, 2] = faku * u[i, j, k, 2]

        # for l in range(3):
        #     FFT991(u[:, :, :, l], work, trigsx, ifaxx, -1, nx, nx, ny * nz)
        #     for k in range(nz):
        #         CFFT99(u[:, :, k, l], work, trigsy, ifaxy, npx // 2, 1, ny, npx // 2, -1)
        #     CFFT99(u[:, :, :, l], work, trigsz, ifaxz, npx // 2 * npy, 1, nz, npx // 2 * npy, -1)
        #     for k in range(nz):
        #         for j in range(ny):
        #             for i in range(nx + 2):
        #                 u[i, j, k, l] = u[i, j, k, l] * facfft
        #                 if i > nx:
        #                     u[i, j, k, l] = 0.

        raise NotImplementedError

    # generate white noise data
    u = np.zeros((nx//2+1, ny, nz, 3), dtype=np.complex128)      #TODO:
    u[:, :, :, :] = 0.
    pi2 = 2*np.pi
    nmax = round(np.sqrt((nx**2 + ny**2 + nz**2))/2)

    E = energy_spectrum_imposed(E, nmax, initial_data, ntopleft, ntopright)
    print(E)

    E = np.array(E)

    for k1 in range(nx // 2):  # 0-15
        i = k1 + 1
        wk1 = abs(k1)
        for k2 in range(-ny // 2 + 1, ny // 2):
            j = (ny + k2) % ny   # 0-15,17-31
            wk2 = k2
            for k3 in range(-nz // 2 + 1, nz // 2):
                k = (nz + k3) % nz   # 0-15,17-31
                wk3 = k3
                wk = np.sqrt(wk1**2 + wk2**2 + wk3**2)
                wk12 = np.sqrt(wk1**2 + wk2**2)
                nk = round(wk)
                amp = np.sqrt(E[nk-1])
                phi1 = pi2 * np.random.random()  # 0.948142290  # np.random.random()  # uniform in [0, 2pi)
                phi2 = pi2 * np.random.random()
                phi3 = pi2 * np.random.random()
                ai = amp * (np.cos(phi1) * np.cos(phi3) + np.sin(phi1) * np.cos(phi3) * 1j)
                bi = amp * (np.cos(phi2) * np.sin(phi3) + np.sin(phi2) * np.sin(phi3) * 1j)
                if wk != 0:
                    if wk12 != 0.:
                        u[i, j, k, 0] = (ai * wk * wk2 + bi * wk1 * wk3) / (wk * wk12)
                        u[i, j, k, 1] = (bi * wk2 * wk3 - ai * wk * wk1) / (wk * wk12)
                    else:
                        u[i, j, k, 0] = ai
                        u[i, j, k, 1] = bi
                    u[i, j, k, 2] = -bi * wk12 / wk
                else:
                    u[i, j, k, 0] = 0.
                    u[i, j, k, 1] = 0.
                    u[i, j, k, 2] = 0.


    ur = np.zeros((nx, ny, nz, 3), dtype=float)
    for l in range(3):
        ur[:,:,:,l] = np.fft.irfftn(u[:, :, :, l], axes=(2,1,0))

    e_spectr = get_energy_spectrum(ur.transpose(3, 0, 1, 2))

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(8,10))
    im = axs[0].imshow(ur[5,:,:,0])
    # im = axs[0].imshow(u[2, :, :,0].real)
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(np.arange(1, nmax+1), E[1:], label='imposed')
    axs[1].plot(np.arange(1, nmax+1), e_spectr[1:nmax+1], label='computed')
    axs[1].legend()
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[1].set_ylim(1e-10, 1e-0)
    fig.savefig('u.png')
    plt.close()






    # u[:, :, :, :] = 0.
    # for k1 in range(nx // 2):
    #     i = k1 * 2 + 1
    #     wk1 = abs(k1)
    #     for k2 in range(-ny // 2 + 1, ny // 2):
    #         j = (ny + k2) % ny + 1
    #         wk2 = k2
    #         for k3 in range(-nz // 2 + 1, nz // 2):
    #             k = (nz + k3) % nz + 1
    #             wk3 = k3
    #             wk = np.sqrt(wk1**2 + wk2**2 + wk3**2)
    #             wk12 = np.sqrt(wk1**2 + wk2**2)
    #             nk = round(wk)
    #             amp = np.sqrt(E[nk-1])
    #             phi1 = pi2 * 0.948142290  # np.random.random()  # uniform in [0, 2pi)
    #             phi2 = pi2 * 0.948142290
    #             phi3 = pi2 * 0.948142290
    #             phi4 = pi2 * 0.948142290
    #             are = amp * np.cos(phi1) * np.cos(phi3)
    #             aim = amp * np.sin(phi1) * np.cos(phi3)
    #             bre = amp * np.cos(phi2) * np.sin(phi3)
    #             bim = amp * np.sin(phi2) * np.sin(phi3)
    #             if wk != 0:
    #                 if wk12 != 0.:
    #                     u[i-1, j-1, k-1, 0] = (are * wk * wk2 + bre * wk1 * wk3) / (wk * wk12)
    #                     u[i, j-1, k-1, 0] = (aim * wk * wk2 + bim * wk1 * wk3) / (wk * wk12)
    #                     u[i-1, j-1, k-1, 1] = (bre * wk2 * wk3 - are * wk * wk1) / (wk * wk12)
    #                     u[i, j-1, k-1, 1] = (bim * wk2 * wk3 - aim * wk * wk1) / (wk * wk12)
    #                 else:
    #                     u[i-1, j-1, k-1, 0] = are
    #                     u[i, j-1, k-1, 0] = aim
    #                     u[i-1, j-1, k-1, 1] = bre
    #                     u[i, j-1, k-1, 1] = bim
    #                 u[i-1, j-1, k-1, 2] = -bre * wk12 / wk
    #                 u[i, j-1, k-1, 2] = -bim * wk12 / wk
    #             else:
    #                 u[i-1, j-1, k-1, 0] = 0.
    #                 u[i, j-1, k-1, 0] = 0.
    #                 u[i-1, j-1, k-1, 1] = 0.
    #                 u[i, j-1, k-1, 1] = 0.
    #                 u[i-1, j-1, k-1, 2] = 0.
    #                 u[i, j-1, k-1, 2] = 0.













    # clean spectrum
    for l in range(3):
        for k in range(nz):
            for j in range(ny):
                u[nx, j, k, l] = 0.
                u[nx+1, j, k, l] = 0.
                u[1, j, k, l] = 0.
        for k in range(nz):
            for i in range(nx):
                u[i, ny//2, k, l] = 0.
        for j in range(ny):
            for i in range(nx):
                u[i, j, nz//2, l] = 0.
        k1 = 0
        i = k1*2+1
        for k2 in range(-ny//2+1, 1):
            kk2 = abs(k2)
            j = (ny+k2) % ny + 1
            jj = (ny+kk2) % ny + 1
            for k3 in range(-nz//2+1, 1):
                kk3 = abs(k3)
                k = (nz+k3) % nz + 1
                kk = (nz+kk3) % nz + 1
                if k2 == -1 or k3 == -1:
                    u[i-1, j-1, k-1, l] = u[i-1, jj-1, kk-1, l]
                    u[i, j-1, k-1, l] = 0.
                    u[i, jj-1, kk-1, l] = 0.
                else:
                    u[i-1, j-1, k-1, l] = u[i-1, jj-1, kk-1, l]
                    u[i, j-1, k-1, l] = -u[i, jj-1, kk-1, l]

    # check divergence
    rmsdiv = 0.
    for k1 in range(nx // 2):
        i = k1 * 2 + 1
        for k2 in range(-ny // 2 + 1, ny // 2):
            j = (ny + k2) % ny + 1
            for k3 in range(-nz // 2 + 1, nz // 2):
                k = (nz + k3) % nz + 1
                wk1 = float(k1)
                wk2 = float(k2)
                wk3 = float(k3)
                rmsxxx = (wk1 * u[i, j, k, 0] + wk2 * u[i, j, k, 1] + wk3 * u[i, j, k, 2])**2 + (
                    wk1 * u[i+1, j, k, 0] + wk2 * u[i+1, j, k, 1] + wk3 * u[i+1, j, k, 2])**2
                rmsdiv = rmsdiv + rmsxxx
    print('rms[div{u(t=0)}] =', np.sqrt(rmsdiv / (nx * ny * nz)))

    # for l in range(3):
    #     for k in range(nz):
    #         for j in range(ny):
    #             for i in range(nx):
    #                 v[i, j, k, l] = u[i, j, k, l]
        # TODO: 
        # CFFT99(v[1,1,1,l], work, trigsz, ifaxz, npx // 2 * npy, 1, nz, npx // 2 * npy, 1)
        # for k in range(nz):
        #     CFFT99(v[1,1,k,l], work, trigsy, ifaxy, npx // 2, 1, ny, npx // 2, 1)
        # FFT991(v[1,1,1,l], work, trigsx, ifaxx, 1, npx, nx, npy*npz, 1)
    for l in range(3):
        v[:, :, :, l] = np.fft.fftn(u[:, :, :, l], axes=(0, 1, 2))

    rmsu = 0.
    rms0 = (v**2).sum()
    # for k in range(nz):
    #     for j in range(ny):
    #         for i in range(nx):
    #             rms0 = rms0 + v[i, j, k, 0]**2 + v[i, j, k, 1]**2 + v[i, j, k, 2]**2
    rmsu = np.sqrt(rms0 / float(nx * ny * nz) / 3.)
    if rmsu == 0.:
        rmsu = 1.
    print(' rmsu = ', rmsu)

    u *= faku / rmsu
    # for k in range(nz):
    #     for j in range(ny):
    #         for i in range(nx):
    #             u[i, j, k, 0] = faku * u[i, j, k, 0] / rmsu
    #             u[i, j, k, 1] = faku * u[i, j, k, 1] / rmsu
    #             u[i, j, k, 2] = faku * u[i, j, k, 2] / rmsu

    # check spectrum  TODO: 
    
    # for l in range(3):
    #     for k in range(nz):
    #         for j in range(ny):
    #             for i in range(nx):
    #                 v[i, j, k, l] = u[i, j, k, l]
    #     CFFT99(v[:, :, :, l], work, trigsz, ifaxz, npx // 2 * npy, 1, nz, npx // 2 * npy, -1)
    #     for k in range(nz):
    #         CFFT99(v[:, :, k, l], work, trigsy, ifaxy, npx // 2, 1, ny, npx // 2, -1)
    #     FFT991(v[:, :, :, l], work, trigsx, ifaxx, 1, npx, nx, npy_npz, -1)

    # for k in range(nz):
    #     for j in range(ny):
    #         for i in range(nx):
    #             for l in range(3):
    #                 u[i, j, k, l] = u[i, j, k, l] / (float(ny) * float(nz))

    # for k in range(nmax + 1):
    #     energ[k] = 0.

    # for k3 in range(-nz // 2, nz // 2):
    #     k = (nz + k3) % nz + 1
    #     wk3 = float(k3)
    #     for k2 in range(-ny // 2, ny // 2):
    #         j = (ny + k2) % ny + 1
    #         wk2 = float(k2)
    #         for k1 in range(-nx // 2, nx // 2):
    #             i = abs(k1) * 2 + 1
    #             wk1 = float(k1)
    #             nk = int(np.sqrt(wk1**2 + wk2**2 + wk3**2))
    #             energ[nk + 1] = energ[nk + 1] + 0.5 * (u[i, j, k, 0]**2 + u[i + 1, j, k, 0]**2 + u[i, j, k, 1]**2 + u[i + 1, j, k, 1]**2 + u[i, j, k, 2]**2 + u[i + 1, j, k, 2]**2)

    # for l in range(3):
    #     CFFT99(u[:, :, :, l], work, trigsz, ifaxz, npx // 2 * npy, 1, nz, npx // 2 * npy, -1)
    #     for k in range(nz):
    #         CFFT99(u[:, :, k, l], work, trigsy, ifaxy, npx // 2, 1, ny, npx // 2, -1)
    #     FFT991(u[:, :, :, l], work, trigsx, ifaxx, 1, npx, nx, npy_npz, -1)
    #     FFT991(u[:, :, :, l], work, trigsx, ifaxx, 1, npx, nx, npy_npz, 1)
    #     for k in range(nz):
    #         CFFT99(u[:, :, k, l], work, trigsy, ifaxy, npx // 2, 1, ny, npx // 2, 1)
    #     CFFT99(u[:, :, :, l], work, trigsz, ifaxz, npx // 2 * npy, 1, nz, npx // 2 * npy, 1)
    #     for k in range(nz):
    #         for j in range(ny):
    #             for i in range(nx + 2):
    #                 u[i, j, k, l] = u[i, j, k, l] / (float(ny) * float(nz))

    # for k in range(nmax + 1):
    #     energ_s[k] = 0.

    # for k3 in range(-nz // 2, nz // 2):
    #     k = (nz + k3) % nz + 1
    #     wk3 = float(k3)
    #     for k2 in range(-ny // 2, ny // 2):
    #         j = (ny + k2) % ny + 1
    #         wk2 = float(k2)
    #         for k1 in range(-nx // 2, nx // 2):
    #             i = abs(k1) * 2 + 1
    #             wk1 = float(k1)
    #             nk = int(np.sqrt(wk1**2 + wk2**2 + wk3**2))
    #             energ_s[nk + 1] = energ_s[nk + 1] + 0.5 * (u[i, j, k, 0]**2 + u[i + 1, j, k, 0]**2 + u[i, j, k, 1]**2 + u[i + 1, j, k, 1]**2 + u[i, j, k, 2]**2 + u[i + 1, j, k, 2]**2)

    # for k in range(nmax + 1):
    #     print(float(k - 1), energ[k], energ_s[k], e[k - 1])

    if cyber[25]:
        raise NotImplementedError
    # if cyber[4]:
    #     if cyber[25]:
    #         concurrent_plot(1, 1)
    # else:
    #     if cyber[25]:
    #         concurrent_plot(1, 1)

    return u

def TINTEG(u):
    # TODO: next steps: TINTEG, RHS_AL, DTSIZE
    global nx, ny, nz, npx, npy, npz
    global cyber, time, itstep
    global tzwstf, deltat

    du = np.zeros((npx, npy, npz, 3))

    global job_id_ctr, taumax, itmax, ioutst, ianalout, irunout, dtauout, tauout_0
    global tauout_1, cfl, tstufe_rq, itstep_rq, cyber
    global job_id_par, reynolds, faku, initial_data, ntopleft, ntopright, captau, itsec
    global isgs_model, alpha_prim, alpha_sec, beta_prim, beta_sec, itype_prim, itype_sec
    global iord_decon, iord_sec, iord_primreg

    stopflag = False
    # time-loop
    for itstep in range(itmax):
        print(f"itstep={itstep}, min(u)={u.min()}, max(u)={u.max()}")
        # end time-stepping
        if time >= taumax:
            stopflag = True

        # Analyse
        icontrol = itstep % ianalout
        if icontrol == 0 or stopflag:
            if cyber[3]:
                # print('ANALYSE at t =', time, 'itstep =', itstep, 'filtered data')
                # ANALYSE(1, 2)
                # print('ANALYSE at t =', time, 'itstep =', itstep, 'de-filtered data')
                # ANALYSE(1, 3)
                raise NotImplementedError("No LES ANALYSE.")
            else:
                print('ANALYSE at t =', time, 'itstep =', itstep)
                print("Ekin = ", 0.5 * (u**2).mean())
                # ANALYSE(1, 1)  # TODO:
                # print('ANALYSE at t =', time, 'itstep =', itstep, 'filtered data')
                # ANALYSE(1, 4)

        # concurrent output
        if cyber[24]:
            raise NotImplementedError('CONCURRENT_PLOT is not implemented')
            # ioutcontrol = itstep % irunout
            # if ioutcontrol == 0:
            #     if cyber[3]:
            #         CONCURRENT_PLOT(1, 2)
            #     else:
            #         CONCURRENT_PLOT(1, 1)

        # field output
        if tauout_0 <= time <= tauout_1:
            if itstep % ioutst == 0 or stopflag:
                # AUSGABE()  # TODO: 
                print('AUSGABE at t =', time)
                print('itstep =', itstep)
                np.save(f'/home/atoshev/code/sphit/data_hit/u_{job_id_ctr}_{job_id_par}_{itstep}.npy', u)
                pass

        # exit time-loop
        if stopflag:
            print('\n*** Zeitlimit erreicht ***')
            break

        # time-stepping
        # TODO: cound not find "deltat" anywhere!! Same goes for "tstufe"
        deltat = DTSIZE(u)

        # Integrationsschritt
        tzwstf = time
        if not cyber[4]:  # all but TGV
            if cyber[1]:
                raise NotImplementedError('RHS is not implemented')
                # RHS(u, du, 0.0)
            else:
                du = RHS_AL(u, du, 0.0)
        else:
            raise NotImplementedError('RHS_NODIV is not implemented')
            # RHS_NODIV(u, du, 0.0)
        u += deltat * du / 3.0
        if cyber[4]:
            raise NotImplementedError('PROJECT is not implemented')
            # PROJECT(u)
        dtau = 1.0
        tzwstf += deltat * dtau / 3.0

        if not cyber[4]:
            if cyber[1]:
                raise NotImplementedError('RHS is not implemented')
                # RHS(u, du, -5.0 / 9.0)
            else:
                du = RHS_AL(u, du, -5/9)
        else:
            raise NotImplementedError('RHS_NODIV is not implemented')
            # RHS_NODIV(u, du, -5.0 / 9.0)
        u += 15.0 * deltat * du / 16.0
        if cyber[4]:
            raise NotImplementedError('PROJECT is not implemented')
            # PROJECT(u)
        dtau = -5.0 / 9.0 * dtau + 1.0
        tzwstf += 15.0 * deltat / 16.0 * dtau

        if not cyber[4]:
            if cyber[1]:
                raise NotImplementedError('RHS is not implemented')
                # RHS(u, du, -153.0 / 128.0)
            else:
                du = RHS_AL(u, du, -153/128)
        else:
            raise NotImplementedError('RHS_NODIV is not implemented')
            # RHS_NODIV(u, du, -153.0 / 128.0)
        u += 8.0 * deltat * du / 15.0
        if cyber[4]:
            raise NotImplementedError('PROJECT is not implemented')
            # PROJECT(u)
        dtau = -153.0 / 128.0 * dtau + 1.0
        tzwstf += 8.0 * deltat / 15.0 * dtau
        time += deltat

    # incremental secondary filter
    if itsec != 0:
        raise NotImplementedError('FILTER2 is not implemented')
        # if itstep % itsec == 0:
        #     print('secondary filtering at', time, itstep)
        #     for l in range(3):
        #         FILTER2(u[0, 0, 0, l])

    # end time stepping
    return 0


def RHS_AL(u, du, rkfac):
    # Spectral computation of incompressible Navier-Stokes RHS without dealiasing
    global nx, ny, nz, npx, npy, npz
    global time, itstep
    global tzwstf, deltat, tauout

    global job_id_ctr, taumax, itmax, ioutst, ianalout, irunout, dtauout, tauout_0
    global tauout_1, cfl, tstufe_rq, itstep_rq, cyber
    global job_id_par, reynolds, faku, initial_data, ntopleft, ntopright, captau, itsec
    global isgs_model, alpha_prim, alpha_sec, beta_prim, beta_sec, itype_prim, itype_sec
    global iord_decon, iord_sec, iord_primreg

    uu = np.zeros((npx, npy, npz, 6))
    ghat = np.zeros((npx, npy, npz))
    ghat2 = np.zeros((npx, npy, npz))
    qhat = np.zeros((npx, npy, npz))
    
    facfft = 1.0 / (ny * nz)

    # Define viscosity
    if reynolds != 0.0:
        reyinv = 1.0 / reynolds
    else:
        reyinv = 0.0
    
    # Roam oddball
    u[nx, :, :, :] = 0.0
    u[nx + 1, :, :, :] = 0.0
    
    # Pseudospectral evaluation of convolution without dealiasing
    v = np.zeros((npx, npy, npz, 3))
    for l in range(3):
        for k3 in range(-nz // 2, nz // 2):
            k = (nz + k3) % nz
            for k2 in range(-ny // 2, ny // 2):
                j = (ny + k2) % ny
                for k1 in range(nx // 2):
                    i = k1 * 2
                    fac0 = qhat[i, j, k]
                    fac1 = qhat[i + 1, j, k]
                    v[i, j, k, l] = fac0 * u[i, j, k, l]
                    v[i + 1, j, k, l] = fac1 * u[i + 1, j, k, l]
        
        # Perform FFTs
        v[:,:,:,l] = np.fft.fftn(v[:,:,:,l], axes=(0, 1, 2))
    
    # Compute uu
    ll = 0
    for l1 in range(1,4):
        for l2 in range(l1, 4):
            ll += 1
            uu[:,:,:,ll-1] = v[:,:,:,l1-1] * v[:,:,:,l2-1]

    # Back to dual space
    for ll in range(6):
        uu[:,:,:,ll] = np.fft.irfftn(uu[:,:,:,ll], axes=(0, 1, 2))
        # uu = np.fft.ifftshift(uu, axes=(0, 1, 2))
        # uu = uu.real * (nx + 2) * (ny + 2) * (nz + 2)
        uu *= facfft  # TODO: check if this is needed
    
    # Assemble RHS
    for k3 in range(-nz // 2, nz // 2):
        k = (nz + k3) % nz
        wk3 = float(k3)
        for k2 in range(-ny // 2, ny // 2):
            j = (ny + k2) % ny
            wk2 = float(k2)
            for k1 in range(nx // 2):
                i = k1 * 2
                wk1 = float(k1)
                wk = wk1 ** 2 + wk2 ** 2 + wk3 ** 2
                wkr = wk * reyinv
                if wk != 0.0:
                    wk11 = wk1 * wk1 / wk
                    wk12 = wk1 * wk2 / wk
                    wk13 = wk1 * wk3 / wk
                    wk22 = wk2 * wk2 / wk
                    wk23 = wk2 * wk3 / wk
                    wk33 = wk3 * wk3 / wk
                else:
                    wk11 = 0.0
                    wk12 = 0.0
                    wk13 = 0.0
                    wk22 = 0.0
                    wk23 = 0.0
                    wk33 = 0.0
                fac0 = ghat[i, j, k]
                fac1 = ghat[i + 1, j, k]
                fac2 = ghat2[i, j, k]
                fac3 = ghat2[i + 1, j, k]
                du[i, j, k, 0] = rkfac * du[i, j, k, 0] - fac0 * (
                    wk1 * wk11 * uu[i + 1, j, k, 0] +
                    2.0 * wk1 * wk12 * uu[i + 1, j, k, 1] +
                    2.0 * wk1 * wk13 * uu[i + 1, j, k, 2] +
                    wk1 * wk22 * uu[i + 1, j, k, 3] +
                    2.0 * wk1 * wk23 * uu[i + 1, j, k, 4] +
                    wk1 * wk33 * uu[i + 1, j, k, 5] -
                    wk1 * uu[i + 1, j, k, 0] -
                    wk2 * uu[i + 1, j, k, 1] -
                    wk3 * uu[i + 1, j, k, 2]
                ) - wkr * u[i, j, k, 0] - captau * (1.0 - fac2) * u[i, j, k, 0]
                du[i + 1, j, k, 0] = rkfac * du[i + 1, j, k, 0] + fac1 * (
                    wk1 * wk11 * uu[i, j, k, 0] +
                    2.0 * wk1 * wk12 * uu[i, j, k, 1] +
                    2.0 * wk1 * wk13 * uu[i, j, k, 2] +
                    wk1 * wk22 * uu[i, j, k, 3] +
                    2.0 * wk1 * wk23 * uu[i, j, k, 4] +
                    wk1 * wk33 * uu[i, j, k, 5] -
                    wk1 * uu[i, j, k, 0] -
                    wk2 * uu[i, j, k, 1] -
                    wk3 * uu[i, j, k, 2]
                ) - wkr * u[i + 1, j, k, 0] - captau * (1.0 - fac3) * u[i + 1, j, k, 0]
                du[i, j, k, 1] = rkfac * du[i, j, k, 1] - fac0 * (
                    wk2 * wk11 * uu[i + 1, j, k, 0] +
                    2.0 * wk2 * wk12 * uu[i + 1, j, k, 1] +
                    2.0 * wk2 * wk13 * uu[i + 1, j, k, 2] +
                    wk2 * wk22 * uu[i + 1, j, k, 3] +
                    2.0 * wk2 * wk23 * uu[i + 1, j, k, 4] +
                    wk2 * wk33 * uu[i + 1, j, k, 5] -
                    wk1 * uu[i + 1, j, k, 1] -
                    wk2 * uu[i + 1, j, k, 3] -
                    wk3 * uu[i + 1, j, k, 4]
                ) - wkr * u[i, j, k, 1] - captau * (1.0 - fac2) * u[i, j, k, 1]
                du[i + 1, j, k, 1] = rkfac * du[i + 1, j, k, 1] + fac1 * (
                    wk2 * wk11 * uu[i, j, k, 0] +
                    2.0 * wk2 * wk12 * uu[i, j, k, 1] +
                    2.0 * wk2 * wk13 * uu[i, j, k, 2] +
                    wk2 * wk22 * uu[i, j, k, 3] +
                    2.0 * wk2 * wk23 * uu[i, j, k, 4] +
                    wk2 * wk33 * uu[i, j, k, 5] -
                    wk1 * uu[i, j, k, 1] -
                    wk2 * uu[i, j, k, 3] -
                    wk3 * uu[i, j, k, 4]
                ) - wkr * u[i + 1, j, k, 1] - captau * (1.0 - fac3) * u[i + 1, j, k, 1]
                du[i, j, k, 2] = rkfac * du[i, j, k, 2] - fac0 * (
                    wk3 * wk11 * uu[i + 1, j, k, 0] +
                    2.0 * wk3 * wk12 * uu[i + 1, j, k, 1] +
                    2.0 * wk3 * wk13 * uu[i + 1, j, k, 2] +
                    wk3 * wk22 * uu[i + 1, j, k, 3] +
                    2.0 * wk3 * wk23 * uu[i + 1, j, k, 4] +
                    wk3 * wk33 * uu[i + 1, j, k, 5] -
                    wk1 * uu[i + 1, j, k, 2] -
                    wk2 * uu[i + 1, j, k, 4] -
                    wk3 * uu[i + 1, j, k, 5]
                ) - wkr * u[i, j, k, 2] - captau * (1.0 - fac2) * u[i, j, k, 2]
                du[i + 1, j, k, 2] = rkfac * du[i + 1, j, k, 2] + fac1 * (
                    wk3 * wk11 * uu[i, j, k, 0] +
                    2.0 * wk3 * wk12 * uu[i, j, k, 1] +
                    2.0 * wk3 * wk13 * uu[i, j, k, 2] +
                    wk3 * wk22 * uu[i, j, k, 3] +
                    2.0 * wk3 * wk23 * uu[i, j, k, 4] +
                    wk3 * wk33 * uu[i, j, k, 5] -
                    wk1 * uu[i, j, k, 2] -
                    wk2 * uu[i, j, k, 4] -
                    wk3 * uu[i, j, k, 5]
                ) - wkr * u[i + 1, j, k, 2] - captau * (1.0 - fac3) * u[i + 1, j, k, 2]
    
    return du


def DTSIZE(u):
    global nx, ny, nz, npx, npy, npz
    global time, itstep
    global tzwstf, deltat, tauout

    global job_id_ctr, taumax, itmax, ioutst, ianalout, irunout, dtauout, tauout_0
    global tauout_1, cfl, tstufe_rq, itstep_rq, cyber
    global job_id_par, reynolds, faku, initial_data, ntopleft, ntopright, captau, itsec
    global isgs_model, alpha_prim, alpha_sec, beta_prim, beta_sec, itype_prim, itype_sec
    global iord_decon, iord_sec, iord_primreg

    # u = np.zeros((npx, npy, npz, 3))
    # work = np.zeros((npx, npy, npz))
    # v = np.zeros((npx, npy, npz, 3))
    # ifaxx = np.zeros(14)
    # ifaxy = np.zeros(14)
    # ifaxz = np.zeros(14)
    # trigsx = np.zeros(2*nx+1)
    # trigsy = np.zeros(2*ny+1)
    # trigsz = np.zeros(2*nz+1)

    tstufe = time
    tau = deltat

    # tau_last = 0.0  # has to be internal variable, i.e. tau from last iteration. See below

    pi = np.pi
    con_2 = 1.0 / reynolds
    pi_delx = 0.5 * float(nx)
    pi_dely = 0.5 * float(ny)
    pi_delz = 0.5 * float(nz)
    if reynolds == 0.0:
        reyinv = 0.0
    else:
        reyinv = 1.0 / reynolds
    
    # for l in range(3):
    #     for k in range(nz):
    #         for j in range(ny):
    #             for i in range(nx):
    #                 v[i, j, k, l] = u[i, j, k, l]
    
    # CFFT99(v[:, :, :, l].flatten(), work.flatten(), trigsz, ifaxz, npx//2*npy, 1, nz, npx//2*npy, 1)
    # for k in range(nz):
    #     CFFT99(v[:, :, k, l].flatten(), work.flatten(), trigsy, ifaxy, npx//2, 1, ny, npx//2, 1)
    # FFT991(v[:, :, :, l].flatten(), work.flatten(), trigsx, ifaxx, 1, npx, nx, npy*npz, 1)
    v = np.zeros_like(u)
    for l in range(3):
        v[:, :, :, l] = np.fft.fftn(u[:, :, :, l], axes=(0, 1, 2))

    umax = np.max(np.abs(v[:,:,:,0]))
    vmax = np.max(np.abs(v[:,:,:,0]))
    wmax = np.max(np.abs(v[:,:,:,0]))
    
    tau = cfl / (umax*pi_delx + vmax*pi_dely + wmax*pi_delz + (pi_delx**2 + pi_dely**2 + pi_delz**2) * reyinv)
    # if tau_last != 0.0:  # TODO: needed?
    #     if abs(tau - tau_last) > 10.0 * abs(tau - tau_last) / tau:
    #         print('tau =', tau)
    #         print('tau varies a lot, consider reducing cfl')
    
    # in our setup, both dtauout = tstufe = 0.0
    if dtauout != 0.0:
        if tstufe + tau > tauout:
            tau = tauout - tstufe
    
    if tstufe + tau > taumax:
        tau = taumax - tstufe

    deltat = tau  # update the global variable
    return tau


if __name__ == "__main__":
    incbox(
        cfg_root="/home/atoshev/code/sphit/py",
        data_root="/home/atoshev/code/sphit/data_hit",
    )
