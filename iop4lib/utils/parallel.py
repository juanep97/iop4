# iop4lib config
import iop4lib.config
iop4conf = iop4lib.Config(config_db=False)

# other imports
import os
import glob
from pathlib import Path
import astropy.io.fits as fits
import psutil
import multiprocessing
import time
import time
import signal
from logging.handlers import QueueHandler
import traceback

# iop4lib imports
from iop4lib.utils import  get_mem_parent_from_child, get_total_mem_from_child, get_mem_current, get_mem_children

# logging
import logging
logger = logging.getLogger(__name__)


# BUILD REDUCED FILES IN MULTIPROCESSING POOL

def epoch_bulkreduce_multiprocesing(reduced_L, epoch=None):
    """ Reduces a list of ReducedFit instances in a multiprocessing pool.

    Invokes the reduction of a list of ReducedFit instances in a multiprocessing pool, with a maximum number of
    concurrent processes defined by iop4conf.nthreads. It can be invoked with a list of ReducedFit 
    from different epochs.

    Parameters
    ----------
    reduced_L : list of ReducedFit
        List of ReducedFit instances to be reduced.

    Other Parameters
    ----------------
    epoch : Epoch, optional
        If provided, it is used only to print the epoch in the log messages of the main thread.
    """

    logger.info(f"{epoch}: starting {iop4conf.nthreads} threads to build {len(reduced_L)} reduced files. Current memory usage: {get_mem_current()/1024**3:.2f} GB.")

    mp_ctx = multiprocessing.get_context('spawn')

    queue = mp_ctx.Manager().Queue(-1)  # create queue listener for multiprocessing logging (-1 so no limit on size)
    listener = logging.handlers.QueueListener(queue, *logging.getLogger().handlers)
    listener.start()
    
    counter = mp_ctx.Value('i', 0)

    with mp_ctx.Pool(processes=iop4conf.nthreads, 
                              initializer=_epoch_bulkreduce_multiprocessing_init,
                              initargs=(counter, queue, len(reduced_L), iop4conf), 
                              maxtasksperchild=20) as pool:

        tasks = [pool.apply_async(_epoch_bulkreduce_multiprocessing_worker, args=(redf,)) for redf in reduced_L]    
    
        pool.close() # no more tasks to be submitted

        _epoch_bulkreduce_multiprocesing_mainloop(tasks, counter, reduced_L, epoch=epoch)
        
        pool.join() # wait for all tasks to finish (should already be finished anyway)
        pool.terminate()
    
    listener.stop()

    logger.info(f"{epoch}: all threads finished. Current memory usage: {get_mem_current()/1024**3:.2f} GB.")



def _epoch_bulkreduce_multiprocessing_init(counter, queue, Nredf, iop4conf):
    """ helper func of bulkreduce_multiprocessing to configure the child processes.
    
    Configures the root logger to send messages to a common queue and to allow only messages 
    from iop4lib (and not from the astrometry solver). iopconf has to be passed as an argument, 
    as if method is spawn, the debug level indicated in the cli is not kept.
    
    Makes _counter (shared) and _Nredf global variables so they can be used by the child processes to keep track of the progress.

    Sets up django ORM so it is usable from the child. 
    """

    # counter and total to show progress

    global _counter
    _counter = counter

    global _Nredf
    _Nredf = Nredf

    # set up logging to send messages to a common queue

    h = QueueHandler(queue)    

    root = logging.getLogger()
    root.handlers.clear()
    iop4conf.configure_logging() # force configure logging 
    root.addHandler(h)
    root.addFilter(logging.Filter("iop4lib"))

    root.setLevel(iop4conf.log_level)

    logger = logging.getLogger(__name__)

    # set up django ORM to be available

    import iop4lib
    iop4conf = iop4lib.Config(**iop4conf)

    # fix for linux where the settings keep being configured even with spawn method
    from django.conf import settings
    if not settings.configured:
        iop4conf.configure_db()

    # close existing connections to the DB
    from django import db
    db.connections.close_all()


class TimeoutException(Exception):
    pass



def _epoch_bulkreduce_multiprocessing_worker_timeout_handler(signum, frame):
    raise TimeoutException



def _epoch_bulkreduce_multiprocessing_worker(reduced_fit: 'ReducedFit'):
    """ helper func to invoke .build() method of ReducedFit on instances from a multiprocessing pool. 

    It needs to be a top-level function, not a method of the class, so it can be pickled and sent to 
    the pool.
    """
    from iop4lib.db import ReducedFit

    logger = logging.getLogger(__name__)
    
    logger.debug(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id} (parent: {get_mem_parent_from_child()/1024**2:.0f} MB, this child: {get_mem_current()/1024**2:.0f} MB, total: {get_total_mem_from_child()/1024**3:.2f} GB total): started building file {_counter.value} / {_Nredf} ({100*_counter.value/_Nredf:.0f}%).")


    try:
        # Start a timer that will send SIGALRM in iop4conf.astrometry_timeout minutes
        signal.signal(signal.SIGALRM, _epoch_bulkreduce_multiprocessing_worker_timeout_handler)
        signal.alarm(iop4conf.astrometry_timeout*60) 
        reduced_fit.build_file()
        signal.alarm(0) # cancel the alarm
    except Exception as e:
        logger.error(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id}: ERROR. Exception was: {e}. Traceback: {traceback.format_exc()}")
    else:
        with _counter.get_lock():
            _counter.value += 1
        logger.debug(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id}: finished building file {_counter.value} / {_Nredf} ({100*_counter.value/_Nredf:.0f}%).")





def _epoch_bulkreduce_multiprocesing_mainloop(tasks, counter, reduced_L, epoch=None):

    time_start = time.time()

    while not all([task.ready() for task in tasks]):

        time_elapsed = time.time() - time_start
        avg_pace = time_elapsed / counter.value if counter.value > 0 else float('inf')

        try:
            childs = [{'pid':child.pid, 'mem':child.memory_info().rss} for child in psutil.Process().children(recursive=True)]
        except psutil.NoSuchProcess:
            # one of the child processes might have dissapeared or restarted in the meantime (because of maxtasksperchild or 
            # because the tasks are close to being finished), so we just ignore it.
            pass
        else:
            logger.info(
                                "\n" + "*"*40 + "\n" +
                                "\n" +
                                (f"[ {epoch} - Main Thread (pid {os.getpid()}) ]\n" if epoch is not None else f"[ Main Thread (pid {os.getpid()}) ]\n") +
                                "\n" +
                                f"Progress: {counter.value} of {len(reduced_L)} files processed in {time_elapsed:.0f} seconds ({avg_pace:.1f} s / file).\n" 
                                "\n" +
                                f"Parent memory: {get_mem_current()/1024**2:5.0f} MB\n" +
                                f"Children memory: {get_mem_children()/1024**3:.2f} GB\n" +
                                "\n".join([f"  - pid {child['pid']:5} : {child['mem']/1024**2:5.0f} MB" for child in childs]) + "\n" +
                                "\n" + 
                                "*"*40
                            )

        time.sleep(5)

    # When task have finished, return
    return


# COMPUTE RELATIVE POLARIMETRY IN MULTIPROCESSING POOL

def _parallel_relative_polarimetry_helper(keys, group):
    from iop4lib.instruments import Instrument
    try:
        Instrument.by_name(keys['instrument']).compute_relative_polarimetry(group)
    except Exception as e:
        logger.error(f"Error for {group=}. Exception {e}: {traceback.format_exc()}")
    finally:
        logger.info(f"Finished computing relative polarimetry for {group=}.")

def parallel_relative_polarimetry(keys, groups):
    mp_ctx = multiprocessing.get_context('fork') # fork is faster and does not need configuring again
    with mp_ctx.Pool(iop4conf.nthreads) as pool:
        pool.starmap(_parallel_relative_polarimetry_helper, zip(keys, groups))