from .scheduler_context import scheduler_context
import dask
import logging
import datetime

import numpy as np

from daskms import xds_from_table, xds_from_ms, xds_to_table, TableProxy

from dask.diagnostics import Profiler, ProgressBar

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) # Add other handlers if you're using this as a library
logger.setLevel(logging.INFO)

#def get_visibility(vis_arr, baselines, i,j):
    #if (i > j):
        #return get_visibility(vis_arr, baselines, j, i)
    
    #return vis_arr[baselines.index([i,j])]

def read_ms(ms, num_vis, res_arcmin, chunks=10000, channel=0):
    '''
        Use dask-ms to load the necessary data to create a telescope operator
        (will use uvw positions, and antenna positions)
        
        -- res_arcmin: Used to calculate the maximum baselines to consider.
                       We want two pixels per smallest fringe
                       pix_res > fringe / 2
                       
                       u sin(theta) = n (for nth fringe)
                       at small angles: theta = 1/u, or u_max = 1 / theta
                       
                       d sin(theta) = lambda / 2
                       d / lambda = 1 / (2 sin(theta))
                       u_max = 1 / 2sin(theta)
                       
                       
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
        logger.info("Antenna Positions {}".format(ant_p.shape))
        
        # Create a dataset representing the field
        field_table = '::'.join((ms, 'FIELD'))
        for field_ds in xds_from_table(field_table):
            #print(ant_ds)
            #print(dask.compute(ant_ds.NAME.data,
                                #ant_ds.POSITION.data, 
                                #ant_ds.DISH_DIAMETER.data))
            phase_dir = np.array(field_ds.PHASE_DIR.data)[0].flatten()
        logger.info("Phase Dir {}".format(np.degrees(phase_dir)))
        
        # Create datasets representing each row of the spw table
        spw_table = '::'.join((ms, 'SPECTRAL_WINDOW'))

        for spw_ds in xds_from_table(spw_table, group_cols="__row__"):
            #print(spw_ds)
            #print(spw_ds.NUM_CHAN.values)
            logger.info("CHAN_FREQ.values: {}".format(spw_ds.CHAN_FREQ.values.shape))
            frequency=dask.compute(spw_ds.CHAN_FREQ.values)[0].flatten()[channel]
            logger.info("Frequency = {}".format(frequency))
            logger.info("NUM_CHAN = %f" % np.array(spw_ds.NUM_CHAN.values)[0])
            wavelength = 2.99793e8 / frequency

        # Create datasets from a partioning of the MS
        datasets = list(xds_from_ms(ms, chunks={'row': chunks}))

        pol = 0
        
        for ds in datasets:
            logger.info("DATA shape: {}".format(ds.DATA.data.shape))
            logger.info("UVW shape: {}".format(ds.UVW.data.shape))
            uvw = np.array(ds.UVW.data)/wavelength   # UVW is stored in meters!
            ant1 = np.array(ds.ANTENNA1.data)
            ant2 = np.array(ds.ANTENNA2.data)
            flags = np.array(ds.FLAG.data)
            cv_vis = np.array(ds.DATA.data)[:,channel,pol]
            epoch_seconds = np.array(ds.TIME.data)[0]
            
            # Try write the STATE_ID column back
            write = xds_to_table(ds, ms, 'STATE_ID')
            with ProgressBar(), Profiler() as prof:
                write.compute()

            # Profile
            #prof.visualize(file_path="chunked.html")
        c = 2.99793e8
        
        # d sin(theta) = \lambda / 2
        theta = np.radians(res_arcmin / 60.0)
        
        u_max = 1.0 / (2 * np.sin(theta))
        logger.info("Resolution Max UVW: {:g}".format(u_max))
        logger.info("Flags: {}".format(flags.shape))

        # Now report the recommended resolution from the data.
        # 1.0 / 2*np.sin(theta) = limit_u
        limit_uvw = np.max(np.abs(uvw), 0)
        res_limit = np.arcsin(1.0 / (2*limit_uvw[0]))
        logger.info("Nyquist resolution: {:g} arcmin".format(np.degrees(res_limit)*60.0))

        if True:
            good_data = np.array(np.where(flags[:,channel,pol] == 0)).T.reshape((-1,))
        else:
            good_data = np.array(np.where((flags[:,channel,pol] == 0) & (np.max(np.abs(uvw), 1) < u_max))).T.reshape((-1,))
        logger.info("Good Data {}".format(good_data.shape))

        logger.info("Maximum UVW: {}".format(limit_uvw))
        logger.info("Minimum UVW: {}".format(np.min(np.abs(uvw), 0)))

        n_ant = len(ant_p)
        
        good_vis = cv_vis[good_data]
        
        n_max = len(good_vis)
        
        indices = np.random.choice(good_data, min(num_vis, n_max))
             
        hdr = {
            'CTYPE1': ('RA---SIN', "Right ascension angle cosine"),
            'CRVAL1': np.degrees(phase_dir)[0],
            'CUNIT1': 'deg     ',
            'CTYPE2': ('DEC--SIN', "Declination angle cosine "),
            'CRVAL2': np.degrees(phase_dir)[1],
            'CUNIT2': 'deg     ',
            'CTYPE3': 'FREQ    ', #           / Central frequency  ",
            'CRPIX3': 1.,
            'CRVAL3': "{}".format(frequency),
            'CDELT3': 10026896.158854,
            'CUNIT3': 'Hz      ',
            'EQUINOX':  '2000.',
            'DATE-OBS': "{}".format(epoch_seconds),
            'BTYPE':   'Intensity'                                                           
        }
        
        #from astropy.wcs.utils import celestial_frame_to_wcs
        #from astropy.coordinates import FK5
        #frame = FK5(equinox='J2010')
        #wcs = celestial_frame_to_wcs(frame)
        #wcs.to_header()

        u_arr = uvw[indices,0]
        v_arr = uvw[indices,1]
        w_arr = uvw[indices,2]
        
        cv_vis = cv_vis[indices]
        
        # Convert from reduced Julian Date to timestamp.
        timestamp = datetime.datetime(1858, 11, 17, 0, 0, 0,
                                      tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=epoch_seconds)

        return u_arr, v_arr, w_arr, cv_vis, hdr, timestamp
        

