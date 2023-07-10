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

# iop4lib imports
from iop4lib.utils import  get_mem_parent_from_child, get_total_mem_from_child, get_mem_current, get_mem_children

# logging
import logging
logger = logging.getLogger(__name__)


# BUILD REDUCED FILES IN MULTIPROCESSING POOL

def epoch_bulkreduce_multiprocesing(reduced_L, epoch=None):
    """ Reduces a list of ReducedFit instances in a multiprocessing pool.

    Invokes the reduction of a list of ReducedFit instances in a multiprocessing pool, with a maximum number of
    concurrent processes defined by iop4conf.max_concurrent_threads. It can be invoked with a list of ReducedFit 
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

    logger.info(f"{epoch}: starting {iop4conf.max_concurrent_threads} threads to build {len(reduced_L)} reduced files. Current memory usage: {get_mem_current()/1024**3:.2f} GB.")

    mp_ctx = multiprocessing.get_context('spawn')

    queue = mp_ctx.Manager().Queue(-1)  # create queue listener for multiprocessing logging (-1 so no limit on size)
    listener = logging.handlers.QueueListener(queue, *logging.getLogger().handlers)
    listener.start()
    
    counter = mp_ctx.Value('i', 0)

    with mp_ctx.Pool(processes=iop4conf.max_concurrent_threads, 
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
    root.addHandler(h)
    root.addFilter(logging.Filter("iop4lib"))

    root.setLevel(iop4conf.log_level)

    logger = logging.getLogger(__name__)

    # set up django ORM to be available

    import iop4lib
    iop4conf = iop4lib.Config(**iop4conf)
    iop4conf.configure_db()

    from django import db
    db.connections.close_all()


class TimeoutException(Exception):
    pass



def _epoch_bulkreduce_multiprocessing_worker_timeout_handler(signum, frame):
    raise TimeoutException



def _epoch_bulkreduce_multiprocessing_worker(reduced_fit):
    """ helper func to invoke .build() method of ReducedFit on instances from a multiprocessing pool. 

    It needs to be a top-level function, not a method of the class, so it can be pickled and sent to 
    the pool.
    """
    from iop4lib.db import ReducedFit

    logger = logging.getLogger(__name__)
    
    logger.debug(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id} (parent: {get_mem_parent_from_child()/1024**2:.0f} MB, this child: {get_mem_current()/1024**2:.0f} MB, total: {get_total_mem_from_child()/1024**3:.2f} GB total): started building file {_counter.value} / {_Nredf} ({100*_counter.value/_Nredf:.0f}%).")


    try:
        # Start a timer that will send SIGALRM in 20 minutes
        signal.signal(signal.SIGALRM, _epoch_bulkreduce_multiprocessing_worker_timeout_handler)
        signal.alarm(20*60) 
        reduced_fit.build_file()
        signal.alarm(0) # cancel the alarm
    except Exception as e:
        logger.error(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id}: ERROR. Exception was: {e}")
    else:
        logger.debug(f"Epoch {reduced_fit.epoch.id}, child pid {os.getpid()}, ReducedFit {reduced_fit.id}: finished building file {_counter.value} / {_Nredf} ({100*_counter.value/_Nredf:.0f}%).")

    with _counter.get_lock():
        _counter.value += 1



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




# BULK REDUCTION IN RAY CLUSTER


def epoch_bulkreduce_ray(reduced_L):
    """Bulk reduction of a list of ReducedFits in a Ray cluster."""
    
    import os
    import ray
    from ray.util.multiprocessing import Pool
    import socket
    from dataclasses import dataclass
    from iop4lib.db import ReducedFit, AstroSource

    logger.info(f"Starting bulk reduction of {len(reduced_L)} ReducedFits in Ray cluster.")

    logger.info("Syncing raw files and DB to Ray cluster.")
    os.system(f"ray rsync-up priv.rayconfig.yaml")
    os.system(fr"rsync -v {iop4conf.db_path} {iop4conf.ray_cluster_address}:'{iop4conf.ray_db_path}'")
    os.system(fr"rsync -va --update {iop4conf.datadir}/raw/ {iop4conf.ray_cluster_address}:'{iop4conf.ray_datadir}/raw/'")
    os.system(fr"rsync -va --update --delete {iop4conf.datadir}/masterflat/ {iop4conf.ray_cluster_address}:'{iop4conf.ray_datadir}/masterflat/'")
    os.system(fr"rsync -va --update --delete {iop4conf.datadir}/masterbias/ {iop4conf.ray_cluster_address}:'{iop4conf.ray_datadir}/masterbias/'")

    logger.info("Connecting to Ray cluster at localhost:25000. Remember to attach to the cluster with 'ray attach priv.rayconfig.yaml -p 25000' and start the head node with 'ray stop' and 'ray start --head --ray-client-server-port 25000 --num-cpus=128'. Additionaly worker nodes can be started with 'ray start --address:head_address:port', port is usually 6379. It might be enough to use ray priv.rayconfig.yaml --restart-only.")
    ray.init("ray://localhost:25000", ignore_reinit_error=True)

    def _init_func():
        import iop4lib
        iop4lib.Config(config_db=True, reconfigure=True)
        import django
        django.db.connections.close_all()
        print(f"Node {socket.gethostname()} initialized.")

    def _buildfile(redf_id):
        """Builds a ReducedFit file in the Ray cluster. 
        
        It syncs the raw files, masterbias and master flat from local, builds the file and performs astrometric calibration remotely,
        and gets the header and summary images back.
        """

        from iop4lib.db import ReducedFit

        print(f"Node {socket.gethostname()} starting to build ReducedFit {redf_id}.")

        try:
            # simplified version of ReducedFit.build_file(), avoids writing to the DB
            redf = ReducedFit.objects.get(id=redf_id)
            redf.apply_masterbias()
            redf.apply_masterflat()
            redf.astrometric_calibration()
        except Exception as e:
            logger.error(f"ReducedFit {redf.id} in Ray cluster -> exception during build_file(): {e}")
            return redf_id, False, None, None
        else:
            #return [redf_id, True, None, None]
        
            res =  [redf_id, True, redf.header, dict()]

            print(f"Node {socket.gethostname()} finished ReducedFit {redf_id}.")

            try:
                for fpath in glob.glob(f"{redf.filedpropdir}/astrometry_*"):
                    with open(fpath, "rb") as f:
                        res[3][os.path.basename(fpath)] = f.read()
            except Exception as e:
                logger.error(f"ReducedFit {redf.id} in Ray cluster -> exception during reading of astrometry plots: {e}")
                return redf_id, False, None, None

            return tuple(res)
           
    
    time_start = time.time()

    with Pool(initializer=_init_func) as pool:

        for i, res in enumerate(pool.imap_unordered(_buildfile, [redf.id for redf in reduced_L], chunksize=4)):

            redf_id, success, header, files = res

            time_elapsed = time.time() - time_start
            avg_pace = time_elapsed / i if i > 0 else float('inf')

            if success:
                logger.info(f"ReducedFit {redf_id} was successfully calibrated astrometrically by the RAY cluster.\nProgress: {i} of {len(reduced_L)} files processed in {time_elapsed:.0f} seconds ({avg_pace:.1f} s / file).")
            else:
                logger.error(f"ReducedFit {redf_id} could not be calibrated astrometrically by the RAY cluster.\nProgress: {i} of {len(reduced_L)} files processed in {time_elapsed:.0f} seconds ({avg_pace:.1f} s / file).")

            redf = ReducedFit.objects.get(id=redf_id)
        
            if not redf.fileexists:
                try:
                    redf.apply_masterbias()
                    redf.apply_masterflat()
                except Exception as e:
                    logger.error(f"{redf}: exception during .apply_masterbias(), .apply_masterflat(): {e}")
                    pass
            
            if success:

                with fits.open(redf.filepath, 'update') as hdul:
                    hdul[0].header.update(header)

                for filename, data in files.items():
                    logger.debug(f"Writting {filename} to {redf.filedpropdir}.")
                    with open(Path(redf.filedpropdir) / filename, "wb") as f:
                        f.write(data)

                redf.sources_in_field.set(AstroSource.get_sources_in_field(fit=redf))

                redf.unset_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)
                redf.set_flag(ReducedFit.FLAGS.BUILT_REDUCED)
                
            else:
                redf.set_flag(ReducedFit.FLAGS.ERROR_ASTROMETRY)
                redf.unset_flag(ReducedFit.FLAGS.BUILT_REDUCED)

            redf.save()

        pool.close() 
        pool.join()
        pool.terminate()


    logger.warning("Not syncing files from Ray cluster! (anyway, all results should have come back already)")
    #os.system(f"rsync -va {iop4conf.ray_cluster_address}:'~/iop4data/calibration' ~/iop4data/calibration")

    n_in_error = ReducedFit.objects.filter(flags__has=ReducedFit.FLAGS.ERROR_ASTROMETRY).count()
    duration_hms_str = time.strftime('%H h %M min %S s', time.gmtime(time.time()-time_start))
    logger.info(f"Finished bulk reduction of {len(reduced_L)} ReducedFits in Ray cluster. {n_in_error} files could not be calibrated astrometrically. Took {duration_hms_str}.")
