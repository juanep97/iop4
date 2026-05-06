import math
import numpy as np
import scipy as sp
import astropy.units as u

import matplotlib as mplt
import matplotlib.pyplot as plt

from iop4lib.typing import *

def get_p_and_chi(q, u, dq, du):
    """Polarization degree (p) and angle (chi) from Stokes (Qr, Ur)."""

    # linear polarization (0 to 1)
    p = math.sqrt(q**2+u**2)
    dp = 1/p * math.sqrt((q*dq)**2 + (u*du)**2)

    # polarization angle (degrees)
    chi = 0.5 * math.degrees(math.atan2(u, q))
    dchi = 0.5 * math.degrees( 1 / (q**2 + u**2) * math.sqrt((q*du)**2 + (u*dq)**2) )

    return p, chi, dp, dchi

def normalize_p_chi(p, chi):
    """Normalizes so that p>0 and -90º < chi < +90º."""
    p = abs(p)
    # chi = ((chi + 90) % 180) - 90
    chi = chi % 180 # if we wanted 0, 180
    return p, chi
    
def apply_corr(Q, U, dQ, dU, Q_inst, U_inst, CPA, dQ_inst, dU_inst, dCPA):
    
    Q_corr = Q - Q_inst
    dQ_corr = math.sqrt(dQ**2 + dQ_inst**2)
    
    U_corr = U - U_inst
    dU_corr = math.sqrt(dU**2 + dU_inst**2)

    p, chi, dp, dchi = get_p_and_chi(Q_corr, U_corr, dQ_corr, dU_corr)

    chi = chi + CPA
    dchi = np.sqrt(dchi**2 + dCPA**2)
    
    return Q_corr, U_corr, p, chi, dp, dchi
    
def eval_model_uncertainty(f, x, popt, pcov, N=1000, s=1):

    samples = np.random.multivariate_normal(popt, pcov, N)
    evaluations = u.Quantity([f(x, *sample) for sample in samples])
    
    lower_bound = np.quantile(evaluations, 1-sp.stats.norm.cdf(s), axis=0)
    upper_bound = np.quantile(evaluations, sp.stats.norm.cdf(s), axis=0)

    return lower_bound, upper_bound

import numpy as np

class Stokes:

    def __init__(self, *args, cov=None, **kwargs):
        
        """
        Flexible Stokes vector container.

        Supported forms:
        1) Stokes(s) where s = (I, q, u)
        2) Stokes(s, cov=...)
        3) Stokes(I=..., q=..., u=..., dI=..., dq=..., du=...)
        4) Stokes(I=..., q=..., u=..., cov=...)
        """

        if args:

            if len(args) == 1:

                s = np.asarray(args[0], dtype=float)

                if s.shape != (3,):
                    raise ValueError("s must be (I, q, u)")
                
                I, q, u = s

            else:

                raise ValueError("Stokes only accepts one or zero positional arguments")

        elif {"I", "q", "u"} <= kwargs.keys():

            I = float(kwargs["I"])
            q = float(kwargs["q"])
            u = float(kwargs["u"])

        else:

            raise ValueError("Must provide either s or I, q, and u")
        
        if cov is not None:

            cov = np.asarray(cov)
            err = np.sqrt(np.diag(cov))
            dI, dq, du = err

        else:

            dI = kwargs.get("dI", 0)
            dq = kwargs.get("dq", 0)
            du = kwargs.get("du", 0)

            cov = np.diag([dI**2, dq**2, du**2])

        self.I = I
        self.q = q
        self.u = u

        self.dI = dI
        self.dq = dq
        self.du = du

        self.cov = cov

        self.compute_p_chi()

        self.p, self.chi = normalize_p_chi(self.p, self.chi)

    def vector(self):
        return np.array([self.I, self.q, self.u])
    
    def __repr__(self):
        return f"Stokes(I={self.I:.6g}, q={self.q:.6g}, u={self.u:.6g})"
    
    def _repr_html_(self):
        return (
            f"{self.__class__.__name__}:<br>\n"
                f" - I: {100*self.I:+.2g} +/- {100*self.dI:.2g}<br>\n"
                f" - q:  ({100*self.q:+.4f} +/- {100*self.dq:.4f} ) % <br>\n"
                f" - p:  ({100*self.u:+.4f} +/- {100*self.du:.4f} ) % <br>\n"
                f" --> p: ({100*self.p:.2f} +/- {100*self.dp:.2f} ) % <br>\n"
                f" --> chi: {self.chi:.2f} +/- {self.dchi:.2f}<br>\n"
        )  
    
    def compute_p_chi(self):

        q = self.q
        u = self.u

        p = math.sqrt(q**2 + u**2)
        chi = 0.5 * math.atan2(u, q)

        J = np.array([
            [q / p,            u / p],
            [-u / (2 * p**2),  q / (2 * p**2)]
        ])

        cov_qu = self.cov[1:, 1:]

        cov_pchi = J @ cov_qu @ J.T

        dp, dchi = np.sqrt(np.diag(cov_pchi))

        chi = math.degrees(chi)
        dchi = math.degrees(dchi)

        self.p = p
        self.dp = dp
        self.chi = chi
        self.dchi = dchi

    def correct(self, q_inst, u_inst, CPA, dq_inst, du_inst, dCPA) -> 'Stokes':

        I, q, u = self.I, self.q, self.u

        I = I
        q = q - q_inst
        u = u - u_inst

        # full transformation

        # J = np.array([
        #     # I, q, u, q_inst, u_inst
        #     [1, 0, 0,  0,  0], # I
        #     [0, 1, 0, -1,  0], # q
        #     [0, 0, 1,  0, -1], # u
        # ])

        # cov = np.zeros((5,5))
        # cov[:3,:3] = self.cov
        # cov[3:, 3:] = np.diag(np.array([dq_inst, du_inst])**2)
        # cov = J @ cov @ J.T
        # cov = cov[:3, :3]

        # faster alternative

        cov = self.cov
        cov[1,1] = cov[1,1] + dq_inst
        cov[2,2] = cov[2,2] + du_inst
        self.cov = cov

        stokes_corr = Stokes((I, q, u), cov=cov)        

        stokes_corr.chi += CPA
        stokes_corr.dchi = np.sqrt(stokes_corr.dchi**2 + dCPA**2)

        corr_p, corr_chi = normalize_p_chi(stokes_corr.p, stokes_corr.chi)
        stokes_corr.p = corr_p
        stokes_corr.chi = corr_chi

        return stokes_corr


def compute_stokes_HWP_analytical(angles, fO, fE, dfO, dfE):

    N = len(angles)
    
    F = (fO - fE) / (fO + fE)
    dF = 2 / ( fO + fE )**2 * np.sqrt(fE**2 * dfO**2 + fO**2 * dfE**2)

    I = (fO + fE)
    dI = np.sqrt(dfO**2 + dfE**2)
    
    q = 2/N * sum([F[i] * math.cos(math.pi/2*i) for i in range(N)])
    dq = 2/N * math.sqrt(sum([dF[i]**2 * math.cos(math.pi/2*i)**2 for i in range(N)]))

    u = 2/N * sum([F[i] * math.sin(math.pi/2*i) for i in range(N)])
    du = 2/N * math.sqrt(sum([dF[i]**2 * math.sin(math.pi/2*i)**2 for i in range(N)]))
    
    sI = np.mean(I)
    dsI = np.std(I)

    s = (sI, q, u)
    ds = (dsI, dq, du)
    scov = np.diag(ds**2)

    stokes = Stokes(s, cov=scov)

    return stokes

def compute_stokes_HWP_fit_full(theta, fO, fE, dfO, dfE, plot=False):

    N = len(theta)
    
    F = (fO - fE) / (fO + fE)
    dF = 2 / ( fO + fE )**2 * np.sqrt(fE**2 * dfO**2 + fO**2 * dfE**2)

    I = (fO + fE)
    dI = np.sqrt(dfO**2 + dfE**2)

    func_fO = lambda theta, I, q, u: 0.5 * ( I + I*q*np.cos(4*theta) + I*u*np.sin(4*theta) )
    func_fE = lambda theta, I, q, u: 0.5 * ( I - I*q*np.cos(4*theta) - I*u*np.sin(4*theta) )

    func =  lambda x, *args: np.concatenate([func_fO(x, *args), func_fE(x, *args)])
    xdata = np.deg2rad(theta)
    ydata = np.concatenate([fO, fE])
    sigma = np.concatenate([dfO, dfE])
    p0 = (np.mean(I), 1, 1)

    popt, pcov, infodict, mesg, ier = sp.optimize.curve_fit(
        func,
        xdata = xdata,
        ydata = ydata,
        sigma = sigma,
        p0 = p0,
        full_output = True,
    )

    # perr = np.sqrt(np.diag(pcov))

    # print(f"{popt=}")
    # print(f"{perr=}")
    # print(f"{pcov=}")
    # print(f"{infodict=}")
    # print(f"{mesg=}")
    # print(f"{ier=}")

    stokes = Stokes(popt, cov=pcov)

    if plot:

        fig, axs = plt.subplots(figsize=(12,12), nrows=4, sharex=True, gridspec_kw=dict(hspace=0))
        
        for i, (func, data_y, data_dy, data_ylabel) in enumerate([
            (func_fO, fO, dfO, 'f_O'),
            (func_fE, fE, dfE, 'f_E'),
        ]):

            # 1st/3rd axes -- fO(fE
            
            ax_idx = 2*i

            axs[ax_idx].errorbar(
                x=theta,
                y=data_y,
                yerr=data_dy,
                linestyle="none",
                marker="o",
                color="k",
                markersize=3,
                label=f"${data_ylabel}$",
            )

            x = np.linspace(min(theta), max(theta), 100)
            y = func(np.deg2rad(x), *popt)
            y_l1s, y_h1s = eval_model_uncertainty(func, np.deg2rad(x), popt, pcov, s=1, N=1000)

            axs[ax_idx].plot(x, y, color="b", linestyle="--", alpha=1)

            axs[ax_idx].fill_between(x, y_l1s, y_h1s,
                color='b',
                alpha=0.1,
                label=r"$1\sigma$",
            )

            axs[ax_idx].set_ylabel(f"${data_ylabel}$")

            # 2nd/4th axis -- fO/fE residuals

            ax_idx = 2*i + 1

            delta_F = data_y - func(np.deg2rad(theta), *popt)
            delta_F_err = data_dy # ignore model uncert?
            
            axs[ax_idx].errorbar(
                x=theta,
                y=delta_F,
                yerr=delta_F_err,
                linestyle="none",
                marker="o",
                color="k",
                markersize=3,
                label=f"${data_ylabel}$",
            )

            axs[ax_idx].fill_between(x, y_l1s-y, y_h1s-y,
                color='r',
                alpha=0.1,
                label=r"$1\sigma$",
            )

            axs[ax_idx].axhline(y=0, color='k', linestyle="-", alpha=0.5)

            axs[ax_idx].set_ylabel(f"Residuals $\\Delta {data_ylabel}$")

        # # annotate with result

        # txt = (
        #     "Results (uncorr instr. pol.)\n"
        #     f"$Q_r$ = {stokes_1[0]:+.4f} $\\pm$ {stokes_err_1[0]:.4f}\n"
        #     f"$U_r$ = {stokes_1[1]:+.4f} $\\pm$ {stokes_err_1[1]:.4f}\n"
        #     f"$p$ = {100*p_2:.2f} $\\pm$ {100*dp_2:.2f} %\n"
        #     f"$\\chi$ = {chi_2:+.2f} $\\pm$ {dchi_2:.2f} º"
        # )

        # text_area = mplt.offsetbox.TextArea(
        #     txt, 
        #     textprops=dict(fontsize=None, ha="left"),
        # )

        # ab = mplt.offsetbox.AnnotationBbox(
        #     text_area,
        #     xy=(0.99, 0.97),
        #     xycoords='axes fraction',
        #     xybox=(0, 0),
        #     boxcoords='offset points',
        #     box_alignment=(1, 1),
        #     frameon=True,
        #     pad=0.3,
        #     bboxprops=dict(
        #         facecolor='white',
        #         edgecolor='gray',
        #         alpha=0.5,
        #         linewidth=1,
        #     ),
        # )

        # axs[0].add_artist(ab)

        plt.show()
        plt.close()

    return stokes

def compute_stokes_HWP_fit_1(theta, fO, fE, dfO, dfE, plot=False):

    N = len(theta)
    
    F = (fO - fE) / (fO + fE)
    dF = 2 / ( fO + fE )**2 * np.sqrt(fE**2 * dfO**2 + fO**2 * dfE**2)

    I = (fO + fE)
    dI = np.sqrt(dfO**2 + dfE**2)

    func_F_qu = lambda theta, q, u: q*np.cos(4*theta) + u*np.sin(4*theta)

    func = func_F_qu
    xdata = np.deg2rad(theta)
    ydata = F
    sigma = dF

    popt, pcov, infodict, mesg, ier = sp.optimize.curve_fit(
        func,
        xdata = xdata,
        ydata = ydata,
        sigma = sigma,
        full_output = True,
    )

    # perr = np.sqrt(np.diag(pcov))

    # print(f"{popt=}")
    # print(f"{perr=}")
    # print(f"{pcov=}")
    # print(f"{infodict=}")
    # print(f"{mesg=}")
    # print(f"{ier=}")

    weights = 1 / dI**2
    sI = np.sum(weights * I) / np.sum(weights)
    dsI = np.sqrt(1 / np.sum(weights))

    s = (sI, *popt)
    scov =np.zeros((3,3))
    scov[0,0] = dsI**2
    scov[1:,1:] = pcov
    
    stokes = Stokes(s, cov=scov)

    if plot:

        fig, axs = plt.subplots(figsize=(12,6), nrows=2, sharex=True, gridspec_kw=dict(hspace=0))

        axs[0].errorbar(
            x=theta,
            y=F,
            yerr=dF,
            linestyle="none",
            marker="o",
            color="k",
            markersize=3,
            label="$F_i$",
        )

        # show fit
        
        x = np.linspace(min(theta), max(theta), 100)
        y = func(np.deg2rad(x), *popt)
        y_l1s, y_h1s = eval_model_uncertainty(func, np.deg2rad(x), popt, pcov, s=1, N=1000)

        axs[0].plot(x, y, color="b", linestyle="--", alpha=1)

        axs[0].fill_between(x, y_l1s, y_h1s,
            color='b',
            alpha=0.1,
            label=r"$1\sigma$",
        )

        # residuals

        delta_F = F - func(np.deg2rad(theta), *popt)
        delta_F_err = dF # ignore model uncert?
        
        axs[1].errorbar(
            x=theta,
            y=delta_F,
            yerr=delta_F_err,
            linestyle="none",
            marker="o",
            color="k",
            markersize=3,
            label="$F_i$",
        )

        axs[1].fill_between(x, y_l1s-y, y_h1s-y,
            color='b',
            alpha=0.1,
            label=r"$1\sigma$",
        )

        axs[1].axhline(y=0, color='k', linestyle="-", alpha=0.5)
        
        # annotate result
        
        txt = (
            "Results (uncorr instr. pol.)\n"
            f"$Q_r$ = {stokes.q:+.4f} $\\pm$ {stokes.dq:.4f}\n"
            f"$U_r$ = {stokes.u:+.4f} $\\pm$ {stokes.du:.4f}\n"
            f"$p$ = {100*stokes.p:.2f} $\\pm$ {100*stokes.dp:.2f} %\n"
            f"$\\chi$ = {stokes.chi:+.2f} $\\pm$ {stokes.dchi:.2f} º"
        )
        
        text_area = mplt.offsetbox.TextArea(
            txt, 
            textprops=dict(fontsize=None, ha="left"),
        )

        ab = mplt.offsetbox.AnnotationBbox(
            text_area,
            xy=(0.99, 0.97),
            xycoords='axes fraction',
            xybox=(0, 0),
            boxcoords='offset points',
            box_alignment=(1, 1),
            frameon=True,
            pad=0.3,
            bboxprops=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=0.5,
                linewidth=1,
            ),
        )

        axs[0].add_artist(ab)

        # title, axes, labels, etc

        axs[-1].set_xticks(theta)
        axs[-1].set_xlabel(r"$\theta_i$ [deg]")
        
        axs[0].set_ylabel("$F_i$")
        axs[1].set_ylabel(r"Residual $\Delta F_i$")
        
        axs[0].legend(loc="upper left")

        fig.tight_layout()
        
        plt.show()
        plt.close()

    return stokes

def compute_stokes_HWP_fit_2(theta, fO=None, dfO=None, fE=None, dfE=None):

    assert bool(fO) ^ bool(fE), "must specify one and only one of fO or fE"
    assert bool(dfO) ^ bool(dfE), "must specify one and only one of dfO or dfE"

    func_fO = lambda theta_i, I, q, u: 0.5 * ( I + I*q*np.cos(4*theta_i) + I*u*np.sin(4*theta_i) )
    func_fE = lambda theta_i, I, q, u: 0.5 * ( I - I*q*np.cos(4*theta_i) - I*u*np.sin(4*theta_i) )

    if fO:
        ydata = fO
        sigma = dfO
        func = func_fO
    else:
        ydata = fE
        sigma = dfE
        func = func_fE

    xdata = np.deg2rad(theta)
    sI0 = 2*np.mean(ydata)
    p0 = (sI0, 1, 1)

    popt, pcov, infodict, mesg, ier = sp.optimize.curve_fit(
        func,
        xdata = xdata,
        ydata = ydata,
        sigma = sigma,
        p0 = p0,
        full_output = True,
    )

    stokes = Stokes(popt, cov=pcov)

    info = {
        'xdata': xdata,
        'xlabel': "\\theta_i",
        'ydata': ydata,
        'ylabel': 'f_O' if fO else 'f_E',
        'f': func,
        'popt': popt,
        'pcov': pcov,
    }

    return stokes, info

def compute_stokes_HWP_fit_3(theta, fO, fE, dfO, dfE):

    N = len(theta)
    
    F = (fO - fE) / (fO + fE)
    dF = 2 / ( fO + fE )**2 * np.sqrt(fE**2 * dfO**2 + fO**2 * dfE**2)

    I = (fO + fE)
    dI = np.sqrt(dfO**2 + dfE**2)

    func_F_pchi = lambda theta_i, p, chi: p * np.cos(4*theta_i - 2*chi)

    func = func_F_pchi
    xdata = np.deg2rad(theta)
    ydata = F
    sigma = dF
    
    popt, pcov, infodict, mesg, ier = sp.optimize.curve_fit(
        func,
        xdata = xdata,
        ydata = ydata,
        sigma = sigma,
        full_output = True,
    )

    perr = np.sqrt(np.diag(pcov))

    weights = 1 / dI**2
    sI = np.sum(weights * I) / np.sum(weights)
    dsI = np.sqrt(1 / np.sum(weights))

    # transform p, chi -> u,q and  the covariance
    p, chi = popt

    q = p * math.cos(2*chi)
    u = p * math.sin(2*chi)

    J = np.array([
        [np.cos(2*chi), -2*p*np.sin(2*chi)],
        [np.sin(2*chi),  2*p*np.cos(2*chi)]
    ])

    scov_qu = J @ pcov @ J.T

    scov = np.zeros((3,3))
    scov[0,0] = dsI**2
    scov[1:,1:] = scov_qu

    s = np.array([sI, q, u])
    
    stokes = Stokes(s, cov=scov)

    return stokes

class PolarimetryGroup(list['ReducedFit']):

    def __repr__(self):

        s = f"<{self.__class__.__name__}>("
        s =  ", ".join([f"{redf.pk}" for redf in self])
        s += ")"
        return s
