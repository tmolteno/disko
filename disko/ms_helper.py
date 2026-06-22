"""
Read data from a measurement set.
Author: Tim Molteno, tim@elec.ac.nz
Copyright (c) 2022-2026.

License. GPLv3.
"""

import datetime
import logging

import numpy as np
from astropy.coordinates import EarthLocation
from casacore.tables import table

from .rime import rayleigh_criterion, resolution_min_baseline

logger = logging.getLogger("disko")


def get_array_location(ms_file: str):
    ms = table(ms_file)
    ant = table(ms.getkeyword("ANTENNA"))
    ant_p = ant.getcol("POSITION")
    ms.unlock()
    ms.close()
    p = np.mean(ant_p, axis=0)
    loc = EarthLocation.from_geocentric(p[0], p[1], p[2], "m")
    geo = loc.to_geodetic(ellipsoid="WGS84")
    logger.info(f"Telescope center : {np.mean(ant_p, axis=0)}")
    logger.info(f"Telescope center : {geo}")
    return {
        "lon": geo.lon.to_value("deg"),
        "lat": geo.lat.to_value("deg"),
        "height": geo.height.to_value("m"),
    }


def casa_read_ms(
    ms_file: str,
    num_vis: int,
    angular_resolution: float,
    ms_column: str = "DATA",
    channel: int = 0,
    field_id: int = 0,
    ddid: int = 0,
    snapshot: int = 0,
    pol: int = 0,
):
    res_deg = angular_resolution  # caller passes degrees from Resolution.degrees()
    logger.info(f"CASA read ms {ms_file} column: {ms_column}")
    ms = table(ms_file, ack=False, readonly=True, lockoptions="auto")

    channel = np.array(channel)
    logger.info(f"Reading channel {channel}")
    logger.debug(f"colnames{ms.colnames()}")
    logger.debug(f"keywordnames{ms.keywordnames()}")
    logger.debug(f"fields{ms.fieldnames()}")

    ant = table(ms.getkeyword("ANTENNA"), ack=False)
    ant_p = ant.getcol("POSITION")
    logger.debug("Antenna Positions {}".format(ant_p.shape))

    # Now use TAQL to select only good data from the correct field
    subt = ms.query(
        f"FIELD_ID=={field_id}",
        sortlist="ARRAY_ID",
        columns=f"TIME, {ms_column}, UVW, ANTENNA1, ANTENNA2, FLAG",
    )

    fields = table(subt.getkeyword("FIELD"), ack=False)
    # field columns ['DELAY_DIR', 'PHASE_DIR', 'REFERENCE_DIR', 'CODE', 'FLAG_ROW', 'NAME', 'NUM_POLY', 'SOURCE_ID', 'TIME']
    phase_dir = fields.getcol("PHASE_DIR")[field_id][0]
    name = fields.getcol("NAME")[field_id]
    field_time = fields.getcol("TIME")[field_id]
    logger.debug(
        f"Field {name} (index {field_id}): Phase Dir {np.degrees(phase_dir)}, t={field_time}"
    )

    times = subt.getcol("TIME")
    logger.debug(f"times {times.shape}")
    time_steps, inverse = np.unique(times, return_inverse=True)
    logger.debug(f"time_steps {time_steps - time_steps[0]}")
    logger.debug(f"inverse {inverse}")

    t_snapshot = time_steps[snapshot]

    snapshot_indices = np.where((inverse == snapshot))[0]
    logger.debug(f"Snapshot {t_snapshot}")
    # logger.debug(f"snapshot_indices {snapshot_indices}")
    logger.debug(f"snapshot_indices {snapshot_indices.shape}")

    uvw = subt.getcol("UVW")
    logger.debug(f"uvw {uvw.shape}")

    ant1 = subt.getcol("ANTENNA1")
    ant2 = subt.getcol("ANTENNA2")
    logger.debug(f"ant = {ant1.shape}")

    flags = subt.getcol("FLAG")
    logger.debug(f"flags = {flags.shape}")
    logger.debug(f"channel = {channel}")

    raw_vis = subt.getcol(ms_column)
    logger.debug(f"raw_vis {raw_vis.shape}")
    raw_vis = raw_vis[snapshot_indices, :, pol][:, channel]
    logger.debug(f"raw_vis {raw_vis.shape}")

    try:
        # Deal with the case where WEIGHT_SPECTRUM is not present.s
        subt_ws = ms.query(
            f"FIELD_ID=={field_id}", sortlist="ARRAY_ID", columns="WEIGHT_SPECTRUM"
        )
        weight_spectrum = subt_ws.getcol("WEIGHT_SPECTRUM")[snapshot_indices, :, pol][
            :, channel
        ]
    except RuntimeError as e:
        logger.debug(f"{e}")
        weight_spectrum = np.ones_like(raw_vis)

    flags = flags[snapshot_indices, :, pol][:, channel]
    uvw = uvw[snapshot_indices, :]
    logger.debug("uvw {}".format(uvw.shape))
    ant1 = ant1[snapshot_indices]
    ant2 = ant2[snapshot_indices]

    # Create datasets representing each row of the spw table
    spw = table(ms.getkeyword("SPECTRAL_WINDOW"), ack=False)
    logger.debug(spw.colnames())

    frequencies = spw.getcol("CHAN_FREQ")[0]

    frequency = frequencies[channel]
    logger.debug(f"Frequencies = {frequencies.shape}")
    logger.debug(f"Frequency = {frequency}")
    logger.debug(f"NUM_CHAN = {np.array(spw.NUM_CHAN[0])}")

    #
    #   Now calculate which indices we should use to get the required number of
    #   visibilities. This is a limit on the baselines to avoid high spatial resolutions
    #   and make our job easier by throwing away some data.
    #
    #   Plan is to use data-sequential inference to calibrate by gradually relaxing this
    #   resolution criterion and using multi-level delayed rejection sampling.
    #
    bl_max = resolution_min_baseline(max_freq=frequency, resolution_deg=res_deg)

    logger.debug("Resolution Max UVW: {:g} meters".format(bl_max))
    logger.debug("Flags: {}".format(flags.shape))
    logger.debug("Snapshot Flags: {}".format(flags.shape))

    # Now report the recommended resolution from the data.
    # 1.0 / 2*np.sin(theta) = limit_u
    limit_uvw = np.max(np.abs(uvw), 0)

    bl = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2 + uvw[:, 2] ** 2)
    res_limit = rayleigh_criterion(max_freq=np.max(frequency), baseline_lengths=bl)
    logger.debug("Nyquist resolution: {:g} arcmin".format(res_limit * 60.0))

    good_data = np.array(np.where((flags == 0) & (bl < bl_max))).T.reshape((-1,))

    logger.debug("Maximum UVW: {}".format(limit_uvw))
    logger.debug("Minimum UVW: {}".format(np.min(np.abs(uvw), 0)))

    for i in range(3):
        p05, p50, p95 = np.percentile(np.abs(uvw[:, i]), [5, 50, 95])
        logger.debug("       U[{}]: {:5.2f} {:5.2f} {:5.2f}".format(i, p05, p50, p95))

    n_max = len(good_data)

    if n_max <= num_vis:
        indices = good_data
    else:
        rng = np.random.default_rng()
        indices = rng.choice(a=good_data, size=num_vis, replace=False)
        indices = np.sort(indices)  # sort to speed up IO

    logger.debug(f"Indices {indices}")

    #
    #   Now read the remaining data
    #
    ant1 = ant1[indices]
    ant2 = ant2[indices]
    weight_spectrum = weight_spectrum[indices]

    raw_vis = raw_vis[indices]
    logger.debug(f"Raw Vis {raw_vis.shape}")

    u_arr = uvw[indices, 0]
    v_arr = uvw[indices, 1]
    w_arr = uvw[indices, 2]
    logger.debug(f"u_arr {u_arr.shape}")

    rms_arr = np.sqrt(1.0 / weight_spectrum)

    logger.debug(f"rms_arr {rms_arr.shape}")

    time = times[indices]
    logger.debug(f"time {time.shape}, timestamp {times[0]}")

    epoch_seconds = np.mean(times)
    logger.debug(f"time {epoch_seconds}, timestamp {times[0]}")

    # Convert from reduced Julian Date to timestamp.
    timestamp = datetime.datetime(
        1858, 11, 17, 0, 0, 0, tzinfo=datetime.timezone.utc
    ) + datetime.timedelta(seconds=epoch_seconds)

    ms.close()

    hdr = {
        "CTYPE1": ("RA---SIN", "Right ascension angle cosine"),
        "CRVAL1": np.degrees(phase_dir)[0],
        "CUNIT1": "deg     ",
        "CTYPE2": ("DEC--SIN", "Declination angle cosine "),
        "CRVAL2": np.degrees(phase_dir)[1],
        "CUNIT2": "deg     ",
        "CTYPE3": "FREQ    ",  # Central frequency
        "CRPIX3": 1.0,
        "CRVAL3": "{}".format(frequency),
        "CDELT3": 10026896.158854,
        "CUNIT3": "Hz      ",
        "EQUINOX": "2000.",
        "DATE-OBS": "{}".format(epoch_seconds),
        "BTYPE": "Intensity",
    }

    # return ant_p, ant1, ant2, u_arr, v_arr, w_arr, frequencies, raw_vis, corrected_vis, seconds, rms_arr
    return u_arr, v_arr, w_arr, frequency, raw_vis, hdr, timestamp, rms_arr, indices
