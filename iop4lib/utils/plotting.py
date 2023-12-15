"""
Some plotting utilities for the IOP4 project.
"""

import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)  

# other imports

import numpy as np
import scipy as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import itertools
from astropy.coordinates import Angle, SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.visualization import SqrtStretch, LogStretch, AsymmetricPercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from pathlib import Path

from iop4lib.utils.sourcepairing import (get_pairs_d, get_pairs_dxy, get_best_pairs)

from .sourcedetection import select_points

# logging

import logging
logger = logging.getLogger(__name__)

def hist_data(data, log=True, ax=None):
                
    if ax is None:
        ax = plt.gca()

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.hist(data, bins="sqrt", log=log, histtype='step')
    
    ax.axvline(x=np.quantile(data, 0.3), color="r")
    ax.axvline(x=np.quantile(data, 0.5), color="g")
    ax.axvline(x=np.quantile(data, 0.99), color="b") 

    ax.xaxis.set_major_locator(mplt.ticker.MaxNLocator(4))

def imshow_w_sources(imgdata, pos1=None, pos2=None, normtype="log", vmin=None, vmax=None, a=10, cmap=None, ax=None, r_aper=20):
    if ax is None:
        ax = plt.gca()

    if isinstance(imgdata, np.ma.masked_array):
        data = imgdata.compressed()
    else:
        data = np.array(imgdata[np.isfinite(imgdata)]).flatten()

    if cmap is None:
        cmap = plt.cm.gray
        cmap.set_bad(color='red')
        cmap.set_under(color='black')
        cmap.set_over(color='white')
            
    if vmin is None:
        vmin = np.quantile(data, 0.3)

    if vmax is None:
        vmax = np.quantile(data, 0.99)

    if normtype == "log":
        #norm = ImageNormalize(imgdata, interval=AsymmetricPercentileInterval(30, 99), stretch=LogStretch(a=10))
        norm = ImageNormalize(imgdata, vmin=vmin, vmax=vmax, stretch=LogStretch(a=a))
        #norm = LogNorm(vmin=vmin, vmax=vmax)
    elif normtype == "logstretch":
        norm = ImageNormalize(stretch=LogStretch(a=a))
    elif normtype == "sqrtstretch":
        norm = ImageNormalize(stretch=SqrtStretch())
    else:
        pass

    ax.imshow(imgdata, cmap=cmap, origin='lower', norm=norm)    

    pos1_present = pos1 is not None and len(pos1) > 0
    pos2_present = pos2 is not None and len(pos2) > 0

    if pos1_present and not pos2_present:
        apertures1 = CircularAperture(pos1, r=r_aper)
        apertures1.plot(color="r", lw=1, alpha=0.9, linestyle='--', ax=ax)
            
    if pos1_present and pos2_present:

        if len(pos1) < 300:
            apertures1 = CircularAperture(pos1, r=r_aper)
            apertures2 = CircularAperture(pos2, r=r_aper)
            
            color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            colors = [next(color_cycle) for _ in range(len(apertures1))]

            for i, (ap1, ap2) in enumerate(zip(apertures1, apertures2)):
                ap1.plot(color=colors[i], lw=1, alpha=0.9, linestyle='--', ax=ax)
                ap2.plot(color=colors[i], lw=1, alpha=0.9, linestyle='-', ax=ax)
        else:
            apertures1 = CircularAperture(pos1, r=r_aper)
            apertures2 = CircularAperture(pos2, r=r_aper)
            
            apertures1.plot(color="m", lw=1, alpha=0.9, linestyle='--', ax=ax)
            apertures2.plot(color="y", lw=1, alpha=0.9, linestyle='-', ax=ax)



def plot_preview_background_substraction_1row(redf, bkg, axs=None, fig=None):

    imgdata_bkg_substracted = redf.mdata - bkg.background

    if axs is None:
        if fig is None:
            fig = plt.gcf()
        axs = fig.subplots(nrows=1, ncols=4)

    axs[0].set_title('Original')
    imshow_w_sources(redf.mdata, ax=axs[0])
    axs[1].set_title('Background')
    imshow_w_sources(bkg.background, ax=axs[1])
    axs[2].set_title('Background-subtracted Data')
    imshow_w_sources(imgdata_bkg_substracted, ax=axs[2])
    axs[3].set_title('Background RMS')
    imshow_w_sources(bkg.background_rms, ax=axs[3])

    
def plot_preview_background_substraction(redf, bkg, axs=None, fig=None):

    if axs is None:
        if fig is None:
            fig = plt.gcf()
        axs = fig.subplots(nrows=2, ncols=4)

    imgdata_bkg_substracted = redf.mdata - bkg.background

    axs[0,0].set_title('Original')
    imshow_w_sources(redf.mdata, ax=axs[0,0])
    hist_data(redf.mdata.compressed(), ax=axs[1,0])
    axs[0,1].set_title('Background')
    imshow_w_sources(bkg.background, ax=axs[0,1])
    hist_data(bkg.background.flatten(), ax=axs[1,1])
    axs[0,2].set_title('Background-subtracted Data')
    imshow_w_sources(imgdata_bkg_substracted, ax=axs[0,2])
    hist_data(imgdata_bkg_substracted.compressed(), ax=axs[1,2])
    axs[0,3].set_title('Background RMS')
    imshow_w_sources(bkg.background_rms, ax=axs[0,3])
    hist_data(bkg.background_rms.flatten(), ax=axs[1,3])


def plot_preview_astrometry(redf, with_simbad=False, legend=True, names_over=False, ax=None, fig=None, **astrocalib_proc_vars):

    from iop4lib.db import AstroSource
    from iop4lib.utils import get_simbad_sources

    def get_matched_sources(match, pos, d_eps=1.412):
        """ Finds sources matching field stars within sqrt(2) pixels. """
        
        wcs = WCS(match.wcs_fields)
        raL = [star.ra_deg*u.deg for star in match.stars]
        decL = [star.dec_deg*u.deg for star in match.stars]
        
        match_coords = SkyCoord(ra=raL, dec=decL)
        matchs_pix = wcs.world_to_pixel(match_coords)

        matched_stars = list()
        
        for i, pix in enumerate(zip(*matchs_pix)):
            d = np.sqrt((pix[0]-pos[:,0])**2 + (pix[1]-pos[:,1])**2)
            if np.min(d) < d_eps:
                matched_stars.append(match_coords[i])
                
        return matched_stars

    ## if wcs1, wcs2 in astrocalib_proc_vars, use them, otherwise use the ones in redf

    if 'wcs1' in astrocalib_proc_vars:
        wcs1 = astrocalib_proc_vars['wcs1']
    else:
        wcs1 = redf.wcs1

    has_pairs = astrocalib_proc_vars.pop('has_pairs', redf.has_pairs)
    
    if has_pairs:
        if 'wcs2' in astrocalib_proc_vars:
            wcs2 = astrocalib_proc_vars['wcs2']
        else:
            wcs2 = redf.wcs2

    ## get detected stars sources in the image
    if 'stars' in astrocalib_proc_vars and astrocalib_proc_vars['stars'] is not None:
        detected_stars = astrocalib_proc_vars['stars']
    else:
        detected_stars = None

    ## get the index stars from the astrometry match 
    if 'solution' in astrocalib_proc_vars:
        indexstars_ra, indexstars_dec = list(zip(*[[star.ra_deg, star.dec_deg] for star in astrocalib_proc_vars['solution'].best_match().stars]))
        indexstars_coords = SkyCoord(ra=indexstars_ra*u.deg,  dec=indexstars_dec*u.deg)
    else:
        indexstars_coords = None

    ## get index stars matching within sqrt(2) pixels of the used centroids.
    if 'bm' in astrocalib_proc_vars and 'stars' in astrocalib_proc_vars:
        matched_stars = get_matched_sources(astrocalib_proc_vars['bm'], np.array(astrocalib_proc_vars['stars']))
    else: 
        matched_stars = None

    ## get sources in field
    sources_in_field = AstroSource.get_sources_in_field(wcs1, width=redf.width, height=redf.height)
    logger.debug(f"{redf}: found {len(sources_in_field)} catalog sources in field: {sources_in_field}")


    if fig is None:
        fig = plt.gcf()

    if ax is None:
        if len(fig.axes) > 0:
            ax = plt.gca()
        else:
            ax = fig.add_subplot(projection=redf.wcs)

    legend_handles_L = list()
    legend_labels_L = list()

    # image

    imshow_w_sources(redf.mdata, ax=ax)

    # # detected stars

    # if detected_stars is not None:
    #     ap = CircularAperture(detected_stars, r=20.0)
    #     h = ap.plot(color="g", lw=1, alpha=1.0, linestyle='--', ax=ax)[-1]
    #     legend_handles_L.append(h)
    #     legend_labels_L.append("detected sources")

    # # index stars

    # if indexstars_coords is not None:
    #     aps = CircularAperture(list(zip(*SkyCoord(indexstars_coords).to_pixel(wcs1))), r=20)
    #     h = aps.plot(color="b", lw=1, alpha=0.8, linestyle='-', ax=ax)[-1]
    #     legend_handles_L.append(h)
    #     legend_labels_L.append("index stars")

    # # matched stars

    # if matched_stars is not None and len(matched_stars) > 0:
    #     aps = CircularAperture(list(zip(*SkyCoord(matched_stars).to_pixel(wcs1))), r=20)
    #     h = aps.plot(color="y", lw=1, alpha=0.8, linestyle='--', ax=ax, label="matched stars")[-1]
    #     legend_handles_L.append(h)
    #     legend_labels_L.append("matched stars (d<sqrt(2))")

    # catalogue sources

    if sources_in_field is not None and len(sources_in_field) > 0:
        for i, source in enumerate(sources_in_field):
            ap = CircularAperture([*source.coord.to_pixel(wcs1)], r=20)
            h = ap.plot(color="r", lw=1, alpha=1, linestyle='-', ax=ax, label=f"{source.name}")
            if has_pairs:
                ax.plot(*source.coord.to_pixel(wcs2), 'rx', alpha=1)
            x, y = source.coord.to_pixel(wcs1)
            ax.annotate(text=source.name if names_over else f"{i}", 
                        xy=(x+20, y), 
                        xytext=(40,0) if names_over else (10,0), 
                        textcoords="offset pixels", 
                        color="red", fontsize=10, weight="bold",
                        verticalalignment="center",
                        #path_effects=[mplt.patheffects.withStroke(linewidth=0.5, foreground="black")],
                        arrowprops=dict(color="red", width=0.5, headwidth=1, headlength=3))
            
            if not names_over:
                legend_handles_L.append(h)
                legend_labels_L.append(f"{i}: {source.name}")

        if names_over:
            legend_handles_L.append(h)
            legend_labels_L.append(f"Catalog sources")

    # overlay sources in Simbad

    if with_simbad:
        try:
            # known sources in this area of sky (Simbad)
            pixscale = np.mean(np.sqrt(np.sum(wcs1.pixel_scale_matrix**2, axis=0)))*u.Unit("deg/pix")
            radius = redf.width * u.Unit('pixel') * pixscale
            centercoord = wcs1.pixel_to_world(redf.mdata.shape[0]//2, redf.mdata.shape[1]//2)

            simbad_sources = get_simbad_sources(centercoord, radius, Nmax=6, exclude_self=False)

            for i, src in enumerate(simbad_sources):
                x, y = src.coord.to_pixel(wcs1)
                ap = CircularAperture([x,y], r=20)
                h = ap.plot(color="yellow", lw=1, alpha=0.8, linestyle='--', ax=ax, label=src.name)
                ax.annotate(text=src.name if names_over else f"{i}", 
                            xy=(x+20, y) if names_over else (x-20, y), 
                            xytext=(40,-20) if names_over else (-15,0), 
                            verticalalignment="center",
                            textcoords="offset pixels", color="yellow", fontsize=10, weight="bold", arrowprops=dict(color='yellow', width=1.0, headwidth=1, headlength=3) if names_over else None)
                if not names_over:
                    legend_handles_L.append(h)
                    legend_labels_L.append(f"{i}: {src.name}")
       
        except Exception as e:
            logger.debug(f"Simbad query failed, ignoring: {e}")

    # Plot the displacement betwen pairs as scale, if specified

    if (disp_sign := astrocalib_proc_vars.get('disp_sign', None)) is not None:
        ax.arrow(x=redf.mdata.shape[0]-20, y=20, dx=disp_sign[0], dy=disp_sign[1], color='red', lw=2, head_width=8, length_includes_head=True)

    # set axes and legends

    if legend:
        if names_over:
            ax.legend(legend_handles_L, legend_labels_L, facecolor=(1,1,1,0.5))
        else:
            ax.legend(legend_handles_L, legend_labels_L, title="Sources", facecolor=(1,1,1,0.5), loc="lower left", bbox_to_anchor=(1,0))

    ax.set_xlim([0, redf.mdata.shape[1]])
    ax.set_ylim([0, redf.mdata.shape[0]])
    ax.coords.grid(True, color='white', ls='solid')
    ax.coords[0].set_axislabel_visibility_rule("always")
    ax.coords[1].set_axislabel_visibility_rule("always")






    



def photopolresult_mplt_viewer(src, band="R", qs0 = None):
    from iop4lib.db import PhotoPolResult, AstroSource
    from astropy.time import Time
    from django.db.models import Q, F, Func, Avg, StdDev, Min, Max

    if isinstance(src, str):
        astrosource = AstroSource.objects.get(name=src)
        srcname = src
    elif isinstance(src, AstroSource):
        astrosource = src
        srcname = astrosource.name
    else:
        raise ValueError("src must be either a string or an AstroSource object")
    
    if qs0 is None:
        qs0 = PhotoPolResult.objects.order_by('-juliandate')

    telescopes = ["OSN-T090", "OSN-T150", "CAHA-T220"]
    colors = ['b', 'g', 'r']

    fig, axs = plt.subplots(nrows=3, figsize=(10,10), sharex=True)
        
    lines_L = list()

    for telescope, color in zip(telescopes, colors):
        qs = qs0.filter(astrosource__name=srcname, epoch__telescope=telescope, band=band)
        
        if qs.count() == 0:
            continue
            
        values_lists = zip(*qs.values_list('juliandate', 'mag', 'mag_err', 'p', 'p_err', 'chi', 'chi_err', 'pk'))
        
        values_lists = [[x if x is not None else np.nan for x in values] for values in values_lists]
        jd, mag, mag_err, p, p_err, chi, chi_err, pk = map(np.array, values_lists)
        datetime = Time(jd, format="jd").datetime

        lines_L.append((0, pk, axs[0].errorbar(x=datetime, y=mag, yerr=mag_err, marker=".", color=color, linestyle="none").lines[0]))
        lines_L.append((1, pk, axs[1].errorbar(x=datetime, y=p, yerr=p_err, marker=".", color=color, linestyle="none").lines[0]))
        lines_L.append((2, pk, axs[2].errorbar(x=datetime, y=chi, yerr=chi_err, marker=".", color=color, linestyle="none").lines[0]))

    # invert magnitude axis
    axs[0].invert_yaxis()

    # secondary axes and its x limits
    datetime_min, datetime_max = mplt.dates.num2date(axs[0].get_xlim())
    mjd_min, mjd_max = Time([datetime_min, datetime_max]).mjd
    ax0_2 = axs[0].twiny()
    ax0_2.set_xlim([mjd_min, mjd_max])

    # x and secondary x labels

    axs[-1].set_xlabel('date')
    ax0_2.set_xlabel('MJD')

    # y labels

    axs[0].set_ylabel(f"{band} mag")
    axs[1].set_ylabel("p")
    axs[1].yaxis.set_major_formatter(mplt.ticker.PercentFormatter(1.0, decimals=1))
    axs[2].set_ylabel("chi [º]")

    # title 
    fig.suptitle(f"{srcname} ({astrosource.other_name})")

    # legend
    legend_handles = [axs[0].plot([],[],color=color, marker=".", linestyle="none", label=telescope)[0] for color, telescope in zip(colors, telescopes)]
    legend_labels = telescopes
    fig.legend(legend_handles, legend_labels, loc="lower left", ncols=3)

    # hover
    qs = PhotoPolResult.objects.order_by('-juliandate').filter(astrosource__name=srcname)

    if qs.count() == 0:
        raise Exception
        
    annots = [ax.annotate("", xy=(0,0), xytext=(-20,20), textcoords="offset points", fontsize=8, bbox=dict(boxstyle="round", fc="gray", pad=0.5), arrowprops=dict(arrowstyle="->")) for ax in axs]

    for annot in annots:
        annot.set_visible(True)

    def update_annots(ax_i, event, line, ind, pks):

        idx = ind["ind"][0]
        pk = pks[idx] 
        obj = PhotoPolResult.objects.get(pk=pk)

        x = Time(obj.juliandate, format="jd").datetime
        y = getattr(obj, ['mag', 'p', 'chi'][ax_i])

        annot = annots[ax_i]
        annot.xy = [x, y]
        print([x,y])

        text = ((f"{obj.epoch.epochname}\n") + 
                (f"id {pk} / {obj.obsmode}\n") + 
                (f"{obj.band}  {obj.mag:.2f} $\\pm$ {obj.mag_err:.2f}\n" if obj.mag is not None else '') + 
                (f"p [%]  {100*obj.p:.1f} $\\pm$ {100*obj.p_err:.1f}\n" if obj.p is not None else '') + 
                (f"chi [º]  {obj.chi:.1f} $\\pm$ {obj.chi_err:.1f}" if obj.chi is not None else ''))
        
        import sys
        print(text, file=sys.__stdout__)
        
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.2)

    def on_plot_hover(event):
        try:
            annots_visible = [False for annot in annots]

            for i, pks, line in lines_L:
                cont, ind = line.contains(event)
                if cont:
                    update_annots(i, event, line, ind, pks)
                    annots_visible[i] = True

            for annot, visible in zip(annots,annots_visible):
                annot.set_visible(visible)

            fig.canvas.draw_idle()
        except Exception as e:
            pass
            #import sys, traceback
            #print(f"{e} {traceback.format_exc()}", file=sys.__stdout__)


    fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)  

    return fig, axs






def build_astrometry_summary_images(redf, astrocalib_proc_vars, summary_kwargs):

    # Summary image of background substraction results

    logger.debug(f"{redf}: plotting astrometry summary image of background substraction results")

    fig = mplt.figure.Figure(figsize=(14,7), dpi=iop4conf.mplt_default_dpi)
    axs = fig.subplots(nrows=2, ncols=4)
    plot_preview_background_substraction(redf, astrocalib_proc_vars['bkg'], axs=axs)
    fig.savefig(Path(redf.filedpropdir) / "astrometry_1_bkgsubstraction.png", bbox_inches="tight")
    fig.clf()

    # Summary images of segmentation and pair finding results

    if 'pos_seg' in astrocalib_proc_vars.keys():

        logger.debug(f"{redf}: plotting astrometry summary image of segmentation results")

        if astrocalib_proc_vars['has_pairs']:
            fig = mplt.figure.Figure(figsize=(12,6), dpi=iop4conf.mplt_default_dpi)
            axs = fig.subplots(nrows=2, ncols=4)

            axs[0,0].set_title('Segmented Image')
            axs[0,0].imshow(astrocalib_proc_vars['segment_map'], origin="lower", cmap=astrocalib_proc_vars['segment_map'].cmap, interpolation="nearest")

            axs[1,0].set_title(f"Kron apertures (N={len(astrocalib_proc_vars['pos_seg'])})")
            imshow_w_sources(astrocalib_proc_vars['imgdata_bkg_substracted'], ax=axs[1,0])
            astrocalib_proc_vars['seg_cat'].plot_kron_apertures(ax=axs[1,0], color='red', lw=1.5)

            axs[0,1].set_title(f"Pairs (n={len(astrocalib_proc_vars['seg1'])}, {len(astrocalib_proc_vars['seg1'])/len(astrocalib_proc_vars['pos_seg'])*100:.1f}%)")
            imshow_w_sources(redf.mdata, pos1=astrocalib_proc_vars['seg1'], pos2=astrocalib_proc_vars['seg2'], ax=axs[0,1])

            axs[1,1].set_title(f"PairsXY (n={len(astrocalib_proc_vars['seg1xy'])}, {len(astrocalib_proc_vars['seg1xy'])/len(astrocalib_proc_vars['pos_seg'])*100:.1f}%)")
            imshow_w_sources(redf.mdata, pos1=astrocalib_proc_vars['seg1xy'], pos2=astrocalib_proc_vars['seg2xy'], ax=axs[1,1])

            axs[0,2].set_title(f"BestPairs (n={len(astrocalib_proc_vars['seg1_best'])}, {len(astrocalib_proc_vars['seg1_best'])/len(astrocalib_proc_vars['pos_seg'])*100:.1f}%)")
            imshow_w_sources(redf.mdata, pos1=astrocalib_proc_vars['seg1_best'], pos2=astrocalib_proc_vars['seg2_best'], ax=axs[0,2])

            axs[1,2].set_title(f"BestPairsXY (n={len(astrocalib_proc_vars['seg1xy_best'])}, {len(astrocalib_proc_vars['seg1xy_best'])/len(astrocalib_proc_vars['pos_seg'])*100:.1f}%)")
            imshow_w_sources(redf.mdata, pos1=astrocalib_proc_vars['seg1xy_best'], pos2=astrocalib_proc_vars['seg2xy_best'], ax=axs[1,2])

            axs[0,3].set_title(f"d0 (1st)")
            get_pairs_d(astrocalib_proc_vars['pos_seg'], d_eps=astrocalib_proc_vars['d_eps'], bins=astrocalib_proc_vars['bins'], hist_range=astrocalib_proc_vars['hist_range'], ax=axs[0,3], doplot=True)
            axs[0,3].set_xlim([0,120])

            get_pairs_dxy(astrocalib_proc_vars['pos_seg'], d_eps=astrocalib_proc_vars['d_eps'], bins=astrocalib_proc_vars['bins'], hist_range=astrocalib_proc_vars['hist_range'], axs=[axs[1,3]], doplot=True)
            axs[1,3].set_title(f"disp (1st)")
            axs[1,3].set_xlim([0,120])

            fig.savefig(Path(redf.filedpropdir) / "astrometry_2_segmentation.png", bbox_inches="tight")
            fig.clf()
        else:
            fig = mplt.figure.Figure(figsize=(6,3), dpi=iop4conf.mplt_default_dpi)
            axs = fig.subplots(nrows=1, ncols=2)
            axs[0].set_title('Segmented Image')
            axs[0].imshow(astrocalib_proc_vars['segment_map'], cmap=astrocalib_proc_vars['segment_map'].cmap, origin="lower", interpolation="nearest")
            axs[1].set_title(f"Kron apertures (N={len(astrocalib_proc_vars['pos_seg'])})")
            imshow_w_sources(astrocalib_proc_vars['imgdata_bkg_substracted'], ax=axs[1])
            astrocalib_proc_vars['seg_cat'].plot_kron_apertures(ax=axs[1], color='red', lw=1.5)

            fig.savefig(Path(redf.filedpropdir) / "astrometry_2_segmentation.png", bbox_inches="tight")
            fig.clf()

    # Summary image of DAOFind and pair finding results

    if 'pos_dao' in astrocalib_proc_vars.keys():

        logger.debug(f"{redf}: plotting astrometry summary image of daofind results")

        if astrocalib_proc_vars['has_pairs']:
            fig = mplt.figure.Figure(figsize=(12,6), dpi=iop4conf.mplt_default_dpi, layout="constrained")
            axs = fig.subplot_mosaic([["A", "B", "C", "D"], 
                                    ["A", "E", "F", "G"]])

            axs["A"].set_title(f"Sources (N={len(astrocalib_proc_vars['pos_dao'])})")
            imshow_w_sources(astrocalib_proc_vars['imgdata_bkg_substracted'], pos1=astrocalib_proc_vars['pos_dao'], ax=axs["A"])

            axs["B"].set_title(f"Pairs (n={len(astrocalib_proc_vars['dao1'])}, {len(astrocalib_proc_vars['dao1'])/len(astrocalib_proc_vars['pos_dao'])*100:.1f}%)")
            imshow_w_sources(astrocalib_proc_vars['imgdata_bkg_substracted'], pos1=astrocalib_proc_vars['dao1'], pos2=astrocalib_proc_vars['dao2'], ax=axs["B"])

            axs["E"].set_title(f"PairsXY (n={len(astrocalib_proc_vars['dao1xy'])}, {len(astrocalib_proc_vars['dao1xy'])/len(astrocalib_proc_vars['pos_dao'])*100:.1f}%)")
            imshow_w_sources(redf.mdata, pos1=astrocalib_proc_vars['dao1xy'], pos2=astrocalib_proc_vars['dao2xy'], ax=axs["E"])

            axs["C"].set_title(f"BestPairs (n={len(astrocalib_proc_vars['dao1_best'])}, {len(astrocalib_proc_vars['dao1_best'])/len(astrocalib_proc_vars['pos_dao'])*100:.1f}%)")
            imshow_w_sources(astrocalib_proc_vars['imgdata_bkg_substracted'], pos1=astrocalib_proc_vars['dao1_best'], pos2=astrocalib_proc_vars['dao2_best'], ax=axs["C"])

            axs["F"].set_title(f"BestPairsXY (n={len(astrocalib_proc_vars['dao1xy'])}, {len(astrocalib_proc_vars['dao1xy_best'])/len(astrocalib_proc_vars['pos_dao'])*100:.1f}%)")
            imshow_w_sources(redf.mdata, pos1=astrocalib_proc_vars['dao1xy_best'], pos2=astrocalib_proc_vars['dao2xy_best'], ax=axs["F"])

            axs['D'].set_title(f"d0 (1st)")
            get_pairs_d(astrocalib_proc_vars['pos_dao'], d_eps=astrocalib_proc_vars['d_eps'], bins=astrocalib_proc_vars['bins'], hist_range=astrocalib_proc_vars['hist_range'], ax=axs['D'], doplot=True)
            axs['D'].set_xlim([0,120])

            get_pairs_dxy(astrocalib_proc_vars['pos_dao'], d_eps=astrocalib_proc_vars['d_eps'], bins=astrocalib_proc_vars['bins'], hist_range=astrocalib_proc_vars['hist_range'], axs=[axs['G']], doplot=True)
            axs['G'].set_title(f"disp (1st)")
            axs['G'].set_xlim([0,120])

            fig.savefig(Path(redf.filedpropdir) / "astrometry_3_daofind.png", bbox_inches="tight")
            fig.clf()
        else:
            fig = mplt.figure.Figure(figsize=(3,3), dpi=iop4conf.mplt_default_dpi)
            ax = fig.subplots(nrows=1, ncols=1)
            ax.set_title(f"Sources (N={len(astrocalib_proc_vars['pos_dao'])})")
            imshow_w_sources(astrocalib_proc_vars['imgdata_bkg_substracted'], pos1=astrocalib_proc_vars['pos_dao'], ax=ax)
            fig.savefig(Path(redf.filedpropdir) / "astrometry_3_daofind.png", bbox_inches="tight")
            fig.clf()

    # Summary image of astrometry results

    if 'wcs1' in astrocalib_proc_vars.keys():
        logger.debug(f"{redf}: plotting astrometry summary image of astrometry results")

        fig = mplt.figure.Figure(figsize=(6,6), dpi=iop4conf.mplt_default_dpi)
        ax = fig.subplots(nrows=1, ncols=1, subplot_kw={'projection': astrocalib_proc_vars['wcs1']})
        plot_preview_astrometry(redf, **astrocalib_proc_vars, ax=ax, fig=fig, with_simbad=summary_kwargs['with_simbad'])
        fig.savefig(Path(redf.filedpropdir) / "astrometry_4_img_result.png", bbox_inches="tight")
        fig.clf()




def plot_finding_chart(target_src, fig=None, ax=None):

    from iop4lib.db import AstroSource
    from iop4lib.utils import get_simbad_sources

    radius = u.Quantity("6 arcmin")

    if fig is None:
        fig = plt.gcf()
    
    if ax is None:
        ax = plt.gca()

    simbad_sources = get_simbad_sources(target_src.coord, radius=radius, Nmax=5, exclude_self=False)
    calibrators = AstroSource.objects.filter(calibrates=target_src)
    
    for src in calibrators:
        ax.plot([src.coord.ra.deg], [src.coord.dec.deg], 'rx', alpha=1)
        ax.annotate(text=src.name, xy=(src.coord.ra.deg+0.001, src.coord.dec.deg), xytext=(+30,0), textcoords="offset pixels", color="red", fontsize=10, weight="bold", verticalalignment="center", horizontalalignment="left", arrowprops=dict(color="red", width=0.5, headwidth=1, headlength=3))
    
    for src in simbad_sources:
        ax.plot([src.coord.ra.deg], [src.coord.dec.deg], 'b+', alpha=1)
        ax.annotate(text=src.name, xy=(src.coord.ra.deg-0.001, src.coord.dec.deg), xytext=(-30,0), textcoords="offset pixels", color="blue", fontsize=10, weight="bold", verticalalignment="center", horizontalalignment="right", arrowprops=dict(color="blue", width=0.5, headwidth=1, headlength=3))

    ax.plot([target_src.coord.ra.deg], [target_src.coord.dec.deg], 'ro', alpha=1)

    # limits (center around target source)

    ax.set_xlim([target_src.coord.ra.deg - radius.to_value("deg"), target_src.coord.ra.deg + radius.to_value("deg")])
    ax.set_ylim([target_src.coord.dec.deg - radius.to_value("deg"), target_src.coord.dec.deg + radius.to_value("deg")])

    # labels

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ax.set_title(f"{target_src.name} ({target_src.other_name})")

    # legend

    target_h = ax.plot([],[], 'ro', label=target_src.name)[0]
    simbad_h = ax.plot([],[], 'b+', label="SIMBAD sources")[0]
    calibrators_h = ax.plot([],[], 'rx', label="IOP4 Calibrators")[0]
    ax.legend(handles=[target_h, calibrators_h, simbad_h], loc="upper right")

    ax.grid(True, color='gray', ls='dashed')

    # secondary axes in hms and dms
    
    lims_ra = ax.get_xlim()
    ax_x2 = ax.twiny()
    ax_x2_ticks = ax.get_xticks()
    ax_x2.set_xticks(ax_x2_ticks)
    ax_x2.set_xticklabels([Angle(x, unit="deg").to_string(unit="hourangle", sep="hms") for x in ax_x2_ticks], fontsize=8)
    ax_x2.set_xlabel("RA [hms]")
    ax_x2.set_xlim(lims_ra)

    lims_dec = ax.get_ylim()
    ax_y2 = ax.twinx()
    ax_y2_ticks = ax.get_yticks()
    ax_y2.set_yticks(ax_y2_ticks)
    ax_y2.set_yticklabels([Angle(x, unit="deg").to_string(unit="deg", sep="dms") for x in ax_y2_ticks], fontsize=8)
    ax_y2.set_ylabel("DEC [dms]")
    ax_y2.set_ylim(lims_dec)