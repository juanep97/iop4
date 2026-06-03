import math
import numpy as np
import scipy as sp
import astropy.units as u
import astropy
from astropy.stats import mad_std

import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, HPacker, VPacker

from iop4lib.typing import *

import logging
logger = logging.getLogger(__name__)

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
    """Normalizes so that p>0 and 0º <= chi <= 180º."""
    p = abs(p)
    chi = chi % 180
    return p, chi
    
def eval_model_uncertainty(f, x, popt, pcov, N=1000, s=1):
    """Returns model's +/- n sigma uncertainty region."""

    samples = np.random.multivariate_normal(popt, pcov, N)
    evaluations = u.Quantity([f(x, *sample) for sample in samples])
    
    lower_bound = np.quantile(evaluations, 1-sp.stats.norm.cdf(s), axis=0)
    upper_bound = np.quantile(evaluations, sp.stats.norm.cdf(s), axis=0)

    return lower_bound, upper_bound

def get_fit_statistics(func, xdata, ydata, sigma, popt, perr, pnames):
    """Returns fit info/statistics."""
    
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    yfit = func(xdata, *popt)
    rres = (ydata - yfit) / sigma

    N = len(ydata)
    k = len(popt)

    chi2 = np.sum(rres**2)
    dof = N - k
    rchi2 = chi2 / dof

    aic = chi2 + 2 * k

    try:
        aicc = aic + (2 * k * (k + 1)) / (N - k - 1)
    except ZeroDivisionError as e:
        aicc = np.nan
    
    bic = chi2 + k * np.log(N)

    return {
        "dof": dof,
        "chi2": chi2,
        "rchi2": rchi2,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "N": N,
        "k": k,
        "popt": popt,
        "perr": perr,
        "pnames": pnames,
    }

def _do_fit(func, xdata, ydata, sigma, p0, bounds=None):
    """Perform curve_fit."""
    
    bounds = bounds if bounds is not None else (-np.inf, +np.inf)

    popt, pcov, infodict, mesg, ier = sp.optimize.curve_fit(
        func,
        xdata = xdata,
        ydata = ydata,
        sigma = sigma,
        p0 = p0,
        bounds = bounds,
        nan_policy = 'omit',
        full_output = True,
    )

    return popt, pcov


def fit_with_sigma_clip(
        func, xdata, ydata, sigma, p0, bounds=None,
        clip_sigma=5, max_iter=3,
        min_keep=4, min_keep_frac=0.74,
        max_discard=2, max_discard_frac=0.3,
        soft=True,
        mode="absmedian",
        get_func=None,
    ):
    """Performs curve_fit with some sigma-clip rejection of outliers."""

    N = len(xdata)
    
    min_keep = max(min_keep, round(min_keep_frac*N))
    max_discard = min(max_discard, round(max_discard_frac*N))

    # print(f"{min_keep=}")
    # print(f"{max_discard=}")

    idx = np.full(N, True)

    for it in range(max_iter):

        # print(f"iteration {it}")

        fit_func = get_func(idx) if get_func else func

        popt, pcov = _do_fit(fit_func, xdata[idx], ydata[idx], sigma[idx], p0, bounds)

        res = ydata - func(xdata, *popt)

        if mode == "zero":
            # measuring dispersion around model (residuals around zero)
            x = res
            cent = 0
            std = np.nanstd(x[idx])
            clip = np.abs(x) > clip_sigma * std
        elif mode == "absmedian":
            # however, if all residuals are far from zero, the std will easily be so
            # high that there is almost no sigma clip. we can measure the dispersion
            # agains how bad they are (around the median of their abs).
            x = np.abs(res)
            cent = np.nanmedian(x[idx])
            std = astropy.stats.mad_std(x[idx])
            clip = np.abs(x) > (cent + clip_sigma * std)
        else:
            raise ValueError

        new_idx = idx & ~clip

        to_keep = sum(new_idx)
        to_discard_total = N - to_keep
        
        # print(f"{x=}")
        # print(f"{cent=}")
        # print(f"{std=}")
        # print(f"{(np.abs(x)-cent)/std=}")
        # print(f"{clip=}")

        # print(f"{to_keep=}")
        # print(f"{to_discard_total=}")

        if std == 0 or not np.isfinite(std):
            break
        
        if soft and to_discard_total > max_discard:

            # up to which one could i discard now?

            can_discard_1 = max_discard - (N-sum(idx))
            can_discard_2 = sum(idx) - min_keep
            can_discard = min(can_discard_1, can_discard_2)

            threshold = np.sort(np.abs(res[idx]))[::-1][can_discard]

            clip = clip & (np.abs(res) > threshold)
            new_idx = idx & ~clip

            to_keep = sum(new_idx)
            to_discard_total = N - to_keep

            # print(f"new {clip=}")
            # print(f"new {to_keep=}")
            # print(f"new {to_discard_total=}")

        # print(f"would keep {to_keep}, discard total {to_discard_total}")

        # enforce constraints

        if to_keep < min_keep:
            # print("min keep reached")
            break

        if to_discard_total > max_discard:
            # print("max discard reached")
            break

        if np.all(new_idx == idx):
            # print("converged")
            break
        
        idx = new_idx
        
        # print(f"it {it}: {to_discard_total=}")
    
    popt, pcov = _do_fit(func, xdata[idx], ydata[idx], sigma[idx], p0)

    # print(f"final {idx=}")

    return popt, pcov, idx

class Stokes:

    def __init__(self, *args, cov=None, **kwargs):
        """Stokes vector container.

        (q,u) are always assumed to be relative to I.

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

    def vector_err(self):
        return np.array([self.dI, self.dq, self.du])
    
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

        # # full transformation

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
        cov[1,1] = cov[1,1] + dq_inst**2
        cov[2,2] = cov[2,2] + du_inst**2
        self.cov = cov

        stokes_corr = Stokes((I, q, u), cov=cov)        

        stokes_corr.chi += CPA
        stokes_corr.dchi = np.sqrt(stokes_corr.dchi**2 + dCPA**2)

        corr_p, corr_chi = normalize_p_chi(stokes_corr.p, stokes_corr.chi)
        stokes_corr.p = corr_p
        stokes_corr.chi = corr_chi

        return stokes_corr



def _build_figure_annotation(fig, fit_stats, stokes, stokes_corr=None, kappa=None, kappa_err=np.nan):

    dof = fit_stats['dof']
    chi2 = fit_stats['chi2']
    rchi2 = fit_stats['rchi2']
    aicc = fit_stats['aicc']
    bic = fit_stats['bic']

    stats_col = VPacker(
        children=[
            TextArea("Fit statistics", textprops=dict(fontsize="large", weight="bold")),
            TextArea(f"$dof$ = {dof}", textprops=dict(fontsize="large")),
            TextArea(f"$\\chi^2$ = {chi2:.2f}", textprops=dict(fontsize="large")),
            TextArea(f"$\\chi^2/dof$ = {rchi2:.2f}", textprops=dict(fontsize="large")),
            TextArea(f"AICC = {aicc:.2f}", textprops=dict(fontsize="large")),
            TextArea(f"BIC = {bic:.2f}", textprops=dict(fontsize="large")),
        ],
        align="left",
        pad=0,
        sep=4,
    )

    results_col = VPacker(
        children=[
            TextArea("Results (uncorr. instr. pol.)", textprops=dict(fontsize="large", weight="bold")),
            TextArea(f"$q$ = ({100*stokes.q:+.2f} ± {100*stokes.dq:.2f})%", textprops=dict(fontsize="large")),
            TextArea(f"$u$ = ({100*stokes.u:+.2f} ± {100*stokes.du:.2f})%", textprops=dict(fontsize="large")),
            TextArea(f"$p$ = ({100*stokes.p:.2f} ± {100*stokes.dp:.2f})%", textprops=dict(fontsize="large")),
            TextArea(f"$\\chi$ = ({stokes.chi:+.2f} ± {stokes.dchi:.2f})º", textprops=dict(fontsize="large")),
        ],
        align="left",
        pad=0,
        sep=4,
    )

    if stokes_corr:
        results_corr_col = VPacker(
            children=[
                TextArea("Results (corrected instr. pol.)", textprops=dict(fontsize="large", weight="bold")),
                TextArea(f"$q$ = ({100*stokes_corr.q:+.2f} ± {100*stokes_corr.dq:.2f})%", textprops=dict(fontsize="large")),
                TextArea(f"$u$ = ({100*stokes_corr.u:+.2f} ± {100*stokes_corr.du:.2f})%", textprops=dict(fontsize="large")),
                TextArea(f"$p$ = ({100*stokes_corr.p:.2f} ± {100*stokes_corr.dp:.2f})%", textprops=dict(fontsize="large")),
                TextArea(f"$\\chi$ = ({stokes_corr.chi:+.2f} ± {stokes_corr.dchi:.2f})º", textprops=dict(fontsize="large")),
            ],
            align="left",
            pad=0,
            sep=4,
        )

    if kappa:
        kappa_col = VPacker(
            children=[
                TextArea("kappa", textprops=dict(fontsize="large", weight="bold")),
                TextArea(f"$\kappa$ = ({100*kappa:+.2f} ± {100*kappa_err:.2f})%", textprops=dict(fontsize="large")),
            ],
            align="left",
            pad=0,
            sep=4,
        )

    annotation_columns = [stats_col, results_col]

    if stokes_corr:
        annotation_columns += [results_corr_col]

    if kappa:
        annotation_columns += [kappa_col]

    box = HPacker(
        children=annotation_columns,
        align="top",
        pad=0,
        sep=30,
    )

    ab = mplt.offsetbox.AnnotationBbox(
        box,
        xy=(0.5, 0.5),
        xycoords=fig.transSubfigure,
        xybox=(0, 0),
        boxcoords='offset points',
        box_alignment=(0.5, 0.5),
        frameon=True,
        pad=0.3,
        bboxprops=dict(
            facecolor='white',
            edgecolor='gray',
            alpha=0.5,
            linewidth=1,
        ),
    )

    return ab

def polmethod(name):
    def wrap(f):
        f.name = name
        return f
    return wrap

@polmethod(name="HWP_analytical")
def compute_stokes_HWP_analytical(
        theta, FO, FE, dFO, dFE,
        inst_pol_dict=None,
        plot=False, fig=None, annotate=False,
    ):
    """Compute polarimetry using an analytical expression ([1]).
    
    References
    ----------
    [1] Ferdinando Patat and Martino Romaniello,
        "Error Analysis for Dual-Beam Optical Linear Polarimetry".
        PASP, 118(839):146-161, January 2006.
        doi:10.1086/497581.
        arXiv:astro-ph/0509153
    """

    N = len(theta)
    
    F = (FO - FE) / (FO + FE)
    dF = 2 / ( FO + FE )**2 * np.sqrt(FE**2 * dFO**2 + FO**2 * dFE**2)

    I = (FO + FE)
    dI = np.sqrt(dFO**2 + dFE**2)
    
    q = 2/N * sum([F[i] * math.cos(math.pi/2*i) for i in range(N)])
    dq = 2/N * math.sqrt(sum([dF[i]**2 * math.cos(math.pi/2*i)**2 for i in range(N)]))

    u = 2/N * sum([F[i] * math.sin(math.pi/2*i) for i in range(N)])
    du = 2/N * math.sqrt(sum([dF[i]**2 * math.sin(math.pi/2*i)**2 for i in range(N)]))
    
    sI = np.mean(I)
    dsI = np.std(I)

    s = np.array([sI, q, u])
    ds = np.array([dsI, dq, du])
    scov = np.diag(ds**2)

    stokes = Stokes(s, cov=scov)

    # we can get some fake fit info for plot

    xdata = np.deg2rad(theta)
    ydata = F
    sigma = dF
    func = lambda theta, q, u: q*np.cos(4*theta) + u*np.sin(4*theta)
    popt = s[1:]
    pcov = scov[1:,1:]
    perr = np.sqrt(np.diag(pcov))
    pnames = ["q", "u"]
    fit_stats = get_fit_statistics(func, xdata, ydata, sigma, popt, perr, pnames)

    if plot:

        fig = fig or plt.figure(figsize=(12,6))
        axs = fig.subplots(nrows=2, sharex=True, gridspec_kw=dict(hspace=0))

        axs[0].errorbar(
            x=theta,
            y=F,
            yerr=dF,
            linestyle="none",
            marker="o",
            color='k',
            markersize=3,
            label="$F_i$",
        )

        # show "fit"

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
        delta_F_err = dF
        
        axs[1].errorbar(
            x=theta,
            y=delta_F,
            yerr=delta_F_err,
            linestyle="none",
            marker="o",
            color='k',
            markersize=3,
            label="$F$",
        )

        axs[1].fill_between(x, y_l1s-y, y_h1s-y,
            color='b',
            alpha=0.1,
            label=r"$1\sigma$",
        )

        axs[1].axhline(y=0, color='k', linestyle="-", alpha=0.5)
        
        if annotate:

            if inst_pol_dict:
                stokes_corr = stokes.correct(**inst_pol_dict)
            else:
                stokes_corr = None
                
            ab = _build_figure_annotation(fig, fit_stats, stokes, stokes_corr=stokes_corr)

            fig.add_artist(ab)
        
        # title, axes, labels, etc

        axs[-1].set_xticks(theta)
        axs[-1].set_xlabel(r"$\theta_i$ [deg]")
        
        axs[0].set_ylabel("$F_i$")
        axs[1].set_ylabel(r"Residual $\Delta F_i$")
        
    return stokes, fit_stats


@polmethod(name="HWP_fit_full")
def compute_stokes_HWP_fit_full(
        theta, FO, FE, dFO, dFE, 
        inst_pol_dict=None,
        plot=False, fig=None, annotate=False,
    ):
    """Compute polarimetry with a fit of (I,q,u) to both the O and the E pairs.
    
    This might perform better when the night is stable (no changes in opacity due
    to clouds passing) and will be more resistant to contamination in only one
    of the pairs.
    """

    N = len(theta)

    func_FO = lambda theta, I, q, u: 0.5 * ( I + I*q*np.cos(4*theta) + I*u*np.sin(4*theta) )
    func_FE = lambda theta, I, q, u: 0.5 * ( I - I*q*np.cos(4*theta) - I*u*np.sin(4*theta) )

    # func =  lambda x, *args: np.concatenate([func_FO(x, *args), func_FE(x, *args)])
    # xdata = np.deg2rad(theta)
    # ydata = np.concatenate([FO, FE])
    # sigma = np.concatenate([dFO, dFE])
    
    # curve_fit supports the above, but to make it easier to mask a single pair,
    # we make it so both input and output have the same length. Since the 
    # function is a concatenation of two functions, it needs to depend on the 
    # mask used.

    xdata = np.repeat(np.deg2rad(theta),2)
    ydata = np.concatenate([FO, FE])
    sigma = np.concatenate([dFO, dFE])

    def get_func(idx):
        s = sum(idx[:N//2])
        def func(x, *args):
            x1 = x[:s]
            x2 = x[s:]
            y1 = func_FO(x1, *args)
            y2 = func_FE(x2, *args)
            y =  np.concatenate([y1, y2])
            return y
        return func
    
    func = get_func(np.full(len(xdata), True))
        
    I0 = (FO + FE)
    p0 = (np.mean(I0), 0.01, 0.01) # ~1% polarization
    bounds_lo = (0.0, -1, -1)
    bounds_hi = (np.inf, +1, +1)
    bounds = (bounds_lo, bounds_hi)

    popt, pcov, idx = fit_with_sigma_clip(func, xdata, ydata, sigma, p0, bounds, get_func=get_func)
    perr = np.sqrt(np.diag(pcov))

    pnames = ["I", "q", "u"]

    fit_stats = get_fit_statistics(get_func(idx), xdata[idx], ydata[idx], sigma[idx], popt, perr, pnames)

    idx = idx.reshape(2,-1) # one row for O another for E

    stokes = Stokes(popt, cov=pcov)
    
    if plot:

        fig = fig or plt.figure(figsize=(12,12))
        axs = fig.subplots(nrows=4, sharex=True, gridspec_kw=dict(hspace=0))
        
        for i, (func, data_y, data_dy, data_ylabel) in enumerate([
            (func_FO, FO, dFO, 'F_O'),
            (func_FE, FE, dFE, 'F_E'),
        ]):
            
            # 1st/3rd axes -- FO/FE
            
            ax_idx = 2*i

            for c, m in [('k', idx), ('r', ~idx)]:
                m = m[i, :]
                axs[ax_idx].errorbar(
                    x=theta[m],
                    y=data_y[m],
                    yerr=data_dy[m],
                    linestyle="none",
                    marker="o",
                    color=c,
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

            axs[ax_idx].set_ylabel(f"${data_ylabel}$ [adu]")

            # 2nd/4th axis -- FO/FE residuals

            ax_idx = 2*i + 1

            delta_F = data_y - func(np.deg2rad(theta), *popt)
            delta_F_err = data_dy # ignore model uncert?
            
            for c, m in [('k', idx), ('r', ~idx)]:
                m = m[i, :]
                axs[ax_idx].errorbar(
                    x=theta[m],
                    y=delta_F[m],
                    yerr=delta_F_err[m],
                    linestyle="none",
                    marker="o",
                    color=c,
                    markersize=3,
                    label=f"${data_ylabel}$",
                )

            axs[ax_idx].fill_between(x, y_l1s-y, y_h1s-y,
                color='r',
                alpha=0.1,
                label=r"$1\sigma$",
            )

            axs[ax_idx].axhline(y=0, color='k', linestyle="-", alpha=0.5)

            axs[ax_idx].set_ylabel(f"Residuals $\\Delta {data_ylabel}$ [adu]")

        if annotate:

            if inst_pol_dict:
                stokes_corr = stokes.correct(**inst_pol_dict)
            else:
                stokes_corr = None
                
            ab = _build_figure_annotation(fig, fit_stats, stokes, stokes_corr=stokes_corr)

            fig.add_artist(ab)

    return stokes, fit_stats

@polmethod(name="HWP_fit_rel")
def compute_stokes_HWP_fit_rel(
        theta, FO, FE, dFO, dFE,
        inst_pol_dict=None,
        plot=False, fig=None, annotate=False,
        kappa=None,
    ):
    """Compute polarimetry with a fit to the relative difference between E and O pairs.
    
    This will usually perform better, specially under certain conditions like
    changes in opacity during observation (e.g. due to passing clouds), but it 
    will be more affected by contamination in any one pair (since a bad data 
    point will weight more).
    """

    N = len(theta)
    
    F = (FO - FE) / (FO + FE)
    dF = 2 / ( FO + FE )**2 * np.sqrt(FE**2 * dFO**2 + FO**2 * dFE**2)

    I = (FO + FE)
    dI = np.sqrt(dFO**2 + dFE**2)

    if kappa is None:
        func_F_qu = lambda theta, q, u: q*np.cos(4*theta) + u*np.sin(4*theta)
    else:
        k = kappa
        func_F_qu = lambda theta, q, u: (k + q*np.cos(4*theta) + u*np.sin(4*theta))/(1 + k*q*np.cos(4*theta) + k*u*np.sin(4*theta))

    func = func_F_qu
    xdata = np.deg2rad(theta)
    ydata = F
    sigma = dF
    p0 = (0.01, 0.01) # ~1% polarization
    bounds_lo = (-1, -1)
    bounds_hi = (+1, +1)
    bounds = (bounds_lo, bounds_hi)

    popt, pcov, idx = fit_with_sigma_clip(func, xdata, ydata, sigma, p0, bounds)
    perr = np.sqrt(np.diag(pcov))

    pnames = ["q", "u"]

    fit_stats = get_fit_statistics(func, xdata, ydata, sigma, popt, perr, pnames)

    # build full stokes and its uncertainty

    weights = 1 / dI**2
    sI = np.sum(weights * I) / np.sum(weights)
    dsI = np.sqrt(1 / np.sum(weights))

    s = (sI, *popt)
    scov = np.zeros((3,3))
    scov[0,0] = dsI**2
    scov[1:,1:] = pcov
    
    stokes = Stokes(s, cov=scov)
    
    if plot:

        fig = fig or plt.figure(figsize=(12,6))
        axs = fig.subplots(nrows=2, sharex=True, gridspec_kw=dict(hspace=0))

        for c, m in [('k', idx), ('r', ~idx)]:
            axs[0].errorbar(
                x=theta[m],
                y=F[m],
                yerr=dF[m],
                linestyle="none",
                marker="o",
                color=c,
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
        
        for c, m in [('k', idx), ('r', ~idx)]:
            axs[1].errorbar(
                x=theta[m],
                y=delta_F[m],
                yerr=delta_F_err[m],
                linestyle="none",
                marker="o",
                color=c,
                markersize=3,
                label="$F$",
            )

        axs[1].fill_between(x, y_l1s-y, y_h1s-y,
            color='b',
            alpha=0.1,
            label=r"$1\sigma$",
        )

        axs[1].axhline(y=0, color='k', linestyle="-", alpha=0.5)
        
        if annotate:

            if inst_pol_dict:
                stokes_corr = stokes.correct(**inst_pol_dict)
            else:
                stokes_corr = None
                
            ab = _build_figure_annotation(fig, fit_stats, stokes, stokes_corr=stokes_corr)

            fig.add_artist(ab)
        
        # title, axes, labels, etc

        axs[-1].set_xticks(theta)
        axs[-1].set_xlabel(r"$\theta_i$ [deg]")
        
        axs[0].set_ylabel("$F_i$")
        axs[1].set_ylabel(r"Residual $\Delta F_i$")
        
    return stokes, fit_stats


@polmethod(name="HWP_fit_rel_nonideal")
def compute_stokes_HWP_fit_rel_nonideal(
        theta, FO, FE, dFO, dFE,
        inst_pol_dict=None,
        plot=False, fig=None, annotate=False,
    ):
    """Compute polarimetry with a fit to the relative difference between E and O pairs (non-ideal HWP case).
    
    Smae as HWP_fit_rel, but allowing deviations from a non-ideal HWP though a 
    free kappa paremter, (see [1] in `compute_stokes_HWP_analytical()`).
    """

    N = len(theta)
    
    F = (FO - FE) / (FO + FE)
    dF = 2 / ( FO + FE )**2 * np.sqrt(FE**2 * dFO**2 + FO**2 * dFE**2)

    I = (FO + FE)
    dI = np.sqrt(dFO**2 + dFE**2)

    func_F_quk = lambda theta, q, u, k: (k + q*np.cos(4*theta) + u*np.sin(4*theta))/(1 + k*q*np.cos(4*theta) + k*u*np.sin(4*theta))

    func = func_F_quk
    xdata = np.deg2rad(theta)
    ydata = F
    sigma = dF
    p0 = (0.01, 0.01, 0.0) # (~1% polarization, ideal HWP)
    bounds_lo = (-1, -1, -1.0)
    bounds_hi = (+1, +1, +1.0)
    bounds = (bounds_lo, bounds_hi)

    popt, pcov, idx = fit_with_sigma_clip(func, xdata, ydata, sigma, p0, bounds)
    perr = np.sqrt(np.diag(pcov))

    pnames = ["q", "u", "k"]

    fit_stats = get_fit_statistics(func, xdata, ydata, sigma, popt, perr, pnames)

    # build full stokes and its uncertainty

    weights = 1 / dI**2
    sI = np.sum(weights * I) / np.sum(weights)
    dsI = np.sqrt(1 / np.sum(weights))

    s = (sI, *popt[:-1])
    scov = np.zeros((3,3))
    scov[0,0] = dsI**2
    scov[1:,1:] = pcov[:-1,:-1]
    
    stokes = Stokes(s, cov=scov)
    
    kappa = popt[-1]
    kappa_err = perr[-1]

    if plot:

        fig = fig or plt.figure(figsize=(12,6))
        axs = fig.subplots(nrows=2, sharex=True, gridspec_kw=dict(hspace=0))

        for c, m in [('k', idx), ('r', ~idx)]:
            axs[0].errorbar(
                x=theta[m],
                y=F[m],
                yerr=dF[m],
                linestyle="none",
                marker="o",
                color=c,
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
        
        for c, m in [('k', idx), ('r', ~idx)]:
            axs[1].errorbar(
                x=theta[m],
                y=delta_F[m],
                yerr=delta_F_err[m],
                linestyle="none",
                marker="o",
                color=c,
                markersize=3,
                label="$F$",
            )

        axs[1].fill_between(x, y_l1s-y, y_h1s-y,
            color='b',
            alpha=0.1,
            label=r"$1\sigma$",
        )

        axs[1].axhline(y=0, color='k', linestyle="-", alpha=0.5)
        
        if annotate:

            if inst_pol_dict:
                stokes_corr = stokes.correct(**inst_pol_dict)
            else:
                stokes_corr = None
                
            ab = _build_figure_annotation(fig, fit_stats, stokes, stokes_corr=stokes_corr, kappa=kappa, kappa_err=kappa_err)

            fig.add_artist(ab)
        
        # title, axes, labels, etc

        axs[-1].set_xticks(theta)
        axs[-1].set_xlabel(r"$\theta_i$ [deg]")
        
        axs[0].set_ylabel("$F_i$")
        axs[1].set_ylabel(r"Residual $\Delta F_i$")
        
    return stokes, fit_stats


@polmethod(name="HWP_fit_1pair")
def compute_stokes_HWP_fit_1pair(
        theta, FO=None, dFO=None, FE=None, dFE=None,
        inst_pol_dict=None,
        plot=False, fig=None, annotate=False,
    ):
    """Compute polarimetry fitting only the O (or E) pair."""

    assert (FO is None) ^ bool(FE is None), "must specify one and only one of FO or FE"
    assert bool(dFO is None) ^ bool(dFE is None), "must specify one and only one of dFO or dFE"

    func_FO = lambda theta_i, I, q, u: 0.5 * ( I + I*q*np.cos(4*theta_i) + I*u*np.sin(4*theta_i) )
    func_FE = lambda theta_i, I, q, u: 0.5 * ( I - I*q*np.cos(4*theta_i) - I*u*np.sin(4*theta_i) )

    if FO:
        ydata = FO
        sigma = dFO
        func = func_FO
        pair = "O"
    else:
        ydata = FE
        sigma = dFE
        func = func_FE
        pair = "E"

    xdata = np.deg2rad(theta)
    sI0 = 2*np.mean(ydata)
    p0 = (sI0, 0.01, 0.01) # ~1% polarization
    bounds_lo = (0.0, -1, -1)
    bounds_hi = (np.inf, +1, +1)
    bounds = (bounds_lo, bounds_hi)

    popt, pcov, idx = fit_with_sigma_clip(func, xdata, ydata, sigma, p0, bounds)
    perr = np.sqrt(np.diag(pcov))

    pnames = ["I", "q", "u"]

    fit_stats = get_fit_statistics(func, xdata, ydata, sigma, popt, perr, pnames)

    stokes = Stokes(popt, cov=pcov)

    if plot:

        fig = fig or plt.figure(figsize=(12,6))
        axs = fig.subplots(nrows=2, sharex=True, gridspec_kw=dict(hspace=0))

        for c, m in [('k', idx), ('r', ~idx)]:
            axs[0].errorbar(
                x=theta[m],
                y=ydata[m],
                yerr=sigma[m],
                linestyle="none",
                marker="o",
                color=c,
                markersize=3,
                label=f"$F_{pair}$",
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

        delta_F = ydata - func(np.deg2rad(theta), *popt)
        delta_F_err = sigma # ignore model uncert?
        
        for c, m in [('k', idx), ('r', ~idx)]:
            axs[1].errorbar(
                x=theta[m],
                y=delta_F[m],
                yerr=delta_F_err[m],
                linestyle="none",
                marker="o",
                color=c,
                markersize=3,
                label=f"$F_{pair}$",
            )

        axs[1].fill_between(x, y_l1s-y, y_h1s-y,
            color='b',
            alpha=0.1,
            label=r"$1\sigma$",
        )

        axs[1].axhline(y=0, color='k', linestyle="-", alpha=0.5)
        
        if annotate:

            if inst_pol_dict:
                stokes_corr = stokes.correct(**inst_pol_dict)
            else:
                stokes_corr = None
                
            ab = _build_figure_annotation(fig, fit_stats, stokes, stokes_corr=stokes_corr)

            fig.add_artist(ab)
        
        # title, axes, labels, etc

        axs[-1].set_xticks(theta)
        axs[-1].set_xlabel(r"$\theta_i$ [deg]")
        
        axs[0].set_ylabel("$F_i$")
        axs[1].set_ylabel(rf"Residual $\Delta F_{pair}$")
            
    return stokes, fit_stats



class PolarimetryGroup(list['ReducedFit']):

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):

        s = f"<{self.__class__.__name__}>("
        s +=  ", ".join([f"{redf.pk}" for redf in self])
        s += ")"
        return s
