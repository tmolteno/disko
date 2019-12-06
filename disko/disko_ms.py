from .disko import DiSkO
from .scheduler_context import scheduler_context
import dask
import logging

import numpy as np

from daskms import xds_from_table, xds_from_ms, xds_to_table, TableProxy

from dask.diagnostics import Profiler, ProgressBar

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)

def get_visibility(vis_arr, baselines, i,j):
    if (i > j):
        return get_visibility(vis_arr, baselines, j, i)
    
    return vis_arr[baselines.index([i,j])]


def disko_from_ms(ms, num_vis, chunks=1000):
    '''
        Use dask-ms to load the necessary data to create a telescope operator
        (will use uvw positions, and antenna positions)
    '''
    with scheduler_context():
        # Create a dataset representing the entire antenna table
        ant_table = '::'.join((ms, 'ANTENNA'))

        wavelength = -1.0;
        for ant_ds in xds_from_table(ant_table):
            #print(ant_ds)
            #print(dask.compute(ant_ds.NAME.data,
                                #ant_ds.POSITION.data, 
                                #ant_ds.DISH_DIAMETER.data))
            ant_p = np.array(ant_ds.POSITION.data)
        logger.info("Antenna Positions %s" % ant_p)
        
        # Create datasets representing each row of the spw table
        spw_table = '::'.join((ms, 'SPECTRAL_WINDOW'))

        for spw_ds in xds_from_table(spw_table, group_cols="__row__"):
            #print(spw_ds)
            #print(spw_ds.NUM_CHAN.values)
            #print(spw_ds.CHAN_FREQ.values)
            frequency=dask.compute(spw_ds.CHAN_FREQ.values)[0]
            logger.info("Frequency = %f" % frequency)
            logger.info("NUM_CHAN = %f" % np.array(spw_ds.NUM_CHAN.values)[0])

        # Create datasets from a partioning of the MS
        datasets = list(xds_from_ms(ms, chunks={'row': chunks}))

        for ds in datasets:
            logger.info("DATA shape: {}".format(ds.DATA.data.shape))
            logger.info("UVW shape: {}".format(ds.UVW.data.shape))
            uvw = np.array(ds.UVW.data)
            ant1 = np.array(ds.ANTENNA1.data)
            ant2 = np.array(ds.ANTENNA2.data)
            cv_vis = np.array(ds.DATA.data)[:,0,0]
            timestamp = np.array(ds.TIME.data)[0]
            
            # Try write the STATE_ID column back
            write = xds_to_table(ds, ms, 'STATE_ID')
            with ProgressBar(), Profiler() as prof:
                write.compute()

            # Profile
            #prof.visualize(file_path="chunked.html")
        c = 2.99793e8
        n_ant = len(ant_p)
        
        n_max = len(cv_vis)
        
        step = max(1, n_max // num_vis)
        
        
        u_arr = uvw[0:n_max:step,0]
        v_arr = uvw[0:n_max:step,1]
        w_arr = uvw[0:n_max:step,2]
        
        cv_vis = cv_vis[0:n_max:step]
        
        ret = DiSkO(u_arr, v_arr, w_arr)
        ret.vis_arr = np.array(cv_vis, dtype=np.complex128)
        ret.timestamp = timestamp
        return ret


