#!/usr/bin/env python
""" iop4_daily_summary.py
    Generate and send a summary of IOP4 results for a given night.

Contact: Juan Escudero Pedrosa (jescudero@iaa.es).
"""

# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=True)

# django imports
from django.template import Context, Template

# iop4lib imports
from iop4lib.db import Epoch, RawFit, ReducedFit, PhotoPolResult, AstroSource
from iop4lib.instruments import Instrument
from iop4lib.enums import SRCTYPES
from iop4lib.utils import get_column_values

# other imports
import os, sys
from pathlib import Path
import argparse
import datetime
import io
import base64
import numpy as np
import scipy as sp
import matplotlib as mplt
import matplotlib.pyplot as plt
from astropy.time import Time
import smtplib, email
from urllib.parse import urlencode, quote_plus

# logging
import coloredlogs, logging
coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)



def gather_context(args):
    r""" Build the context for the night summary template."""

    context = Context()
    context['night'] = args.date

    # List epochs for the given date
    # annotate the number of distinct sources in each epoch
    
    epochs = list(Epoch.objects.filter(night=args.date))

    # annotate epochs with some useful information

    for epoch in epochs:
        # annotate with the set of (non-calibrator) sources in the results each epoch

        epoch.srcname_list = list(PhotoPolResult.objects.filter(epoch=epoch)
                                  .exclude(astrosource__in=AstroSource.objects.filter(is_calibrator=True))
                                  .values_list("astrosource__name", flat=True)
                                  .distinct())
        
        # annotate with the list of files with ERROR_ASTROMETRY

        epoch.files_with_error_astrometry = list(ReducedFit.objects.filter(epoch=epoch)
                                                  .filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY))
        epoch.files_with_error_astrometry_ids_str_L = [str(reducedfit.id) for reducedfit in epoch.files_with_error_astrometry]
        epoch.files_with_error_astrometry_href = args.site_url + "/iop4/admin/iop4api/reducedfit/?id__in=" + ",".join(epoch.files_with_error_astrometry_ids_str_L)
        # TODO: include IOP4Admin in Django configure and use reverse as in PhotoPolResult admin for line above

    # get list of all sources for which there are band R results in this night
        
    sources = AstroSource.objects.exclude(is_calibrator=True).filter(
        photopolresults__epoch__night=args.date, 
        photopolresults__band="R",
    ).distinct()

    # check if some of these sources lack calibrators
    sources_without_calibrators = [source for source in sources if not source.calibrators.exists()]

    if sources:
        instruments = [instrument.name for instrument in Instrument.get_known()]
        colors = [mplt.colormaps['tab10'](i) for i in range(len(instruments))]

    results_summary_images = dict()

    for source in sources:

        logger.info(f"Plotting summary of results for source {source.name}")

        # check which was the previous night with results for this source

        prev_night = (PhotoPolResult.objects
                      .filter(astrosource=source, band="R", epoch__night__lt=args.date)
                      .order_by('-epoch__night')
                      .values_list('epoch__night', flat=True)
                      .first())

        # Get the results for the source in the given night, but if there is only
        # one result, get the results from the previous night for comparison

        qs_today = PhotoPolResult.objects.filter(astrosource=source, band="R").filter(epoch__night=args.date)

        if prev_night is not None and qs_today.count() == 1:
            qs0 = PhotoPolResult.objects.filter(astrosource=source, band="R").filter(epoch__night__gte=prev_night, epoch__night__lte=args.date).order_by('-juliandate')
        else:
            qs0 = qs_today.order_by('-juliandate')


        fig = mplt.figure.Figure(figsize=(1000/100, 600/100), dpi=100)
        axs = fig.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'hspace': 0.05})

        for instrument, color in zip(instruments, colors):
            qs = qs0.filter(instrument=instrument)

            if not qs.exists():
                continue

            column_names = ['id', 'juliandate', 'band', 'instrument', 'mag', 'mag_err', 'p', 'p_err', 'chi', 'chi_err', 'flags']
            vals =  get_column_values(qs, column_names)
            vals['datetime'] = Time(vals['juliandate'], format='jd').datetime

            axs[0].errorbar(x=vals['datetime'], y=vals['mag'], yerr=vals['mag_err'], marker=".", color=color, linestyle="none")
            axs[1].errorbar(x=vals['datetime'], y=vals['p'], yerr=vals['p_err'], marker=".", color=color, linestyle="none")
            axs[2].errorbar(x=vals['datetime'], y=vals['chi'], yerr=vals['chi_err'], marker=".", color=color, linestyle="none")

        # x-axis date locator and formatter
        from matplotlib.dates import AutoDateLocator
        axs[-1].xaxis.set_major_locator(AutoDateLocator(interval_multiples=False, maxticks=7))

        # invert magnitude axis
        axs[0].invert_yaxis()

        # secondary axes and its x limits
        datetime_min, datetime_max = mplt.dates.num2date(axs[0].get_xlim())
        mjd_min, mjd_max = Time([datetime_min, datetime_max]).mjd
        ax0_2 = axs[0].twiny()
        ax0_2.set_xlim([mjd_min, mjd_max])
        ax0_2.ticklabel_format(useOffset=False)

        # major x and secondary x ticks
        ax0_2.xaxis.set_major_locator(mplt.ticker.MaxNLocator(5))
        
        # x and secondary x labels
        axs[-1].set_xlabel('date')
        ax0_2.set_xlabel('MJD')

        # y labels
        axs[0].set_ylabel(f"mag (R)")
        axs[1].set_ylabel("p")
        axs[1].yaxis.set_major_formatter(mplt.ticker.PercentFormatter(1.0, decimals=1))
        axs[2].set_ylabel("chi [º]")

        # title 
        fig.suptitle(f"{source.name} ({source.other_names})" if source.other_names else source.name)

        # legend
        legend_handles = [axs[0].plot([],[],color=color, marker=".", linestyle="none", label=instrument)[0] for color, instrument in zip(colors, instruments)]
        fig.legend(handles=legend_handles, ncols=3, loc='upper left', bbox_to_anchor=(0, 0))

        # save the figure to a buffer and encode it to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.clf()
        buf.seek(0)
        imgbytes = buf.read()
        imgb64 = base64.b64encode(imgbytes).decode("utf-8")

        # build the link to the interactive source plot

        url_args = dict()

        url_args['srcname'] = source.name

        if prev_night is not None and qs_today.count() == 1:
            url_args['from'] = str(prev_night)
        else:
            url_args['from'] = (qs0.last().datetime - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        # we need 12:00 of the next day
        next_day_noon = (datetime.datetime.combine(args.date, datetime.time(12, 0)) + datetime.timedelta(days=1))
        url_args['to'] = str(next_day_noon)

        source_plot_url = args.site_url + "/iop4/explore/plot/?" + urlencode(url_args, quote_via=quote_plus)

        # save the results to the context

        results_summary_images[source.name] = dict(imgb64=imgb64, source_plot_url=source_plot_url)

    # save vars to context and return it
    
    if args.log_file:
        context['log_url'] = args.site_url + "/iop4/explore/logs/?" + urlencode({'log_file':Path(args.log_file).name}, quote_via=quote_plus)

    context['epochs'] = epochs    
    context['sources'] = sources
    context['results_summary_images'] = results_summary_images
    context['sources_without_calibrators'] = sources_without_calibrators
    context['args'] = args

    return context
 


def generate_html_summary(args, context):
    """ Render the summary template with the given context."""
    
    with open(Path(__file__).parent / 'iop4_night_summary.html', 'r') as f:
        template = Template(f.read())
        return template.render(context=context)



def send_email(args, summary_html):
    """ Send the html summary by email."""

    msg = email.message.EmailMessage()

    msg['Subject'] = f"IOP4 summary {args.date}"
    msg['From'] = args.fromaddr 
    msg['To'] = args.mailto
    msg['reply-to'] = args.contact_email
    msg.set_content(summary_html, subtype="html")

    logger.debug("Sending email")

    with smtplib.SMTP('localhost') as s:
        s.send_message(msg)



def main():
    argparser = argparse.ArgumentParser(description='IOP4 night summary',
                                        prog="iop4-night-summary",
                                        epilog = __doc__,
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        allow_abbrev=False)
    
    argparser.add_argument('--date', type=str, default=None, help='Date of the night in format YYYY-MM-DD (default: yesterday)')
    argparser.add_argument('--mailto', type=lambda s: s.split(','), default=None, help='One or several email addresses separated by commas')
    argparser.add_argument('--fromaddr', type=str, default="iop4@localhost", help='Email address of the sender')
    argparser.add_argument('--contact-name', type=str, default=None, help='Name to indicate as contact, if any')
    argparser.add_argument('--contact-email', type=str, default=None, help='Email to indicate as contact (default is the sender address)')
    argparser.add_argument('--site-url', type=str, default="localhost:8000", help='URL of the site to link the summary')
    argparser.add_argument('--log-file', type=str, default=None, help='IOP4 log file')
    argparser.add_argument('--saveto', type=str, default=None, help='Save the summary to a file')
    argparser.add_argument('--rc', type=int, default=None, help="Indicate the return code from IOP4 to warn the user if the data processing fails")

    args = argparser.parse_args()

    if args.date is None:
        args.date = datetime.date.today() - datetime.timedelta(days=1)
    else:
        try:
            args.date = datetime.datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print('Invalid date format. Use YYYY-MM-DD.')
            sys.exit(-1)

    if args.contact_email is None:
        args.contact_email = args.fromaddr
    
    logger.info(f'Generating daily summary for {args.date}')
    context = gather_context(args)   

    logger.info('Rendering html summary')
    html_summary = generate_html_summary(args, context)

    if args.mailto:
        logger.info('Sending email')
        send_email(args, html_summary) 
    else:
        logger.warning('No email address provided.')

    if args.saveto:
        logger.info(f'Saving summary to {args.saveto}')
        with open(args.saveto, 'w') as f:
            f.write(html_summary)
    else:
        logger.warning('No file provided to save the summary.')

    logger.info('Done')

    sys.exit(0)

if __name__ == '__main__':
    main()
