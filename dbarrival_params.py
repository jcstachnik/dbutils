import sys,os

from numpy import array 
import numpy as np
import argparse,logging,time
from multiprocessing.pool import Pool
from functools import partial
import math
from datetime import timedelta
import pdb

sys.path.append( os.environ['ANTELOPE'] + '/data/python' )

# import antelope stuff first
import antelope.datascope as ds
import antelope.stock as stock

from obspy import Stream,Trace,UTCDateTime
from obspy.signal.polarization import polarization_analysis

import matplotlib.pyplot as plt

def setup_logger(name='root',loglevel='WARNING',logfile='/dev/null'):
    '''

    Quick setup for logging 
    logger = setup_logger(name='mylogger',loglevel='DEBUG',logfile='mylog.txt')

    '''
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging,loglevel))
    # Set up logging to the console
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging,loglevel))
    # Set up logging to a file
    fh = logging.FileHandler(logfile,mode='w')
    fh.setLevel(logging.DEBUG)
    FORMAT = '%(levelname)-8s: %(message)s'
    formatter = logging.Formatter(FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.debug('Logging turned on at level {0}'.format(loglevel))
    return logger

def getargs():
    """
    Get the command line arguments
    """

    parser = argparse.ArgumentParser(description=
    '''
    This script calculates some arrival parameters
    ''',
    epilog='''
    Example:
    db3polar.py -d indb -o outdb -s "orid==1814"

    TODO command line arrival subset
    TODO deal with accelerometers
    TODO multiproc the db loop
    ''',
    formatter_class=argparse.RawTextHelpFormatter)

    helpmsg = "Input database name."
    parser.add_argument("-d", "--dbname", required=True, type=str, help=helpmsg)
    helpmsg = "Output database name."
    parser.add_argument("-o", "--odbname", required=True, 
                            type=str,
                            help=helpmsg)
    helpmsg = "Origin table subset."
    parser.add_argument("-s", "--osub", required=False, 
                            type=str,
                            help=helpmsg)
    helpmsg = "Arrival-assoc join table subset."
    parser.add_argument("-a", "--asub", required=False, 
                            type=str,
                            help=helpmsg)
    helpmsg = "Number of processes to run at a time."
    parser.add_argument("-n", "--nproc", required=False, 
                            type=int, default=4,
                            help=helpmsg)
    helpmsg = "Do pre-subset before loadchans."
    parser.add_argument("-p", "--presub", required=False, 
                            action='store_true', default=False,
                            help=helpmsg)
    helpmsg = "Operate dryrun mode. Set this flag to exit after db loop, no calculations."
    parser.add_argument("--dryrun", required=False, 
                            action='store_true', default=False,
                            help=helpmsg)
    helpmsg = "If set, calculate baz based on polarization."
    parser.add_argument("--dopolar", required=False, 
                            action='store_true', default=False,
                            help=helpmsg)
    helpmsg = "If set, calculate first motion."
    parser.add_argument("--dofm", required=False, 
                            action='store_true', default=False,
                            help=helpmsg)
    helpmsg = "If set, calculate SNR ."
    parser.add_argument("--dosnr", required=False, 
                            action='store_true', default=False,
                            help=helpmsg)

    parser.add_argument("-l", "--log", dest="logLevel", default='INFO',
                choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                help="Set the logging level for the console. [INFO]")
    helpmsg = "Logfile for logging output. Default is just to console." \
              "Logging level is set to DEBUG for the output file."
    parser.add_argument("-L","--logfile", type=str, required=False, default='/dev/null',
                help=helpmsg)

    args = parser.parse_args()
    return args,parser

def sort_chans(chans):
    """
    A utility function to sort channel codes into Z, N, E or Z, 1, 2
    order.
    """
    sorted_chans = []
    for chan in chans:
        if chan[2] == "Z":
            sorted_chans += [chan]
    for chan in chans:
        if chan[2] == "N" or chan[2] == "1":
            sorted_chans += [chan]
    for chan in chans:
        if chan[2] == "E" or chan[2] == "2":
            sorted_chans += [chan]
    return sorted_chans

def get_chan3(dbsite_chan, sta, chan, time):
    """
    Return the appropriate 3 channel tuple for 3 component
    stations.
    """
    logger = logging.getLogger('db3polar')
    scsub = "sta =~ /{}/ && chan =~ /{}.{}/ "\
                    "&& ondate < _{}_ && (offdate > _{}_ || "\
                    "offdate == -1)".format(sta,
                                            chan[:2],
                                            chan[3:],
                                            time,
                                            time)
    logger.debug('get_chan3: scsub {}'.format(scsub))
    view = dbsite_chan.subset("sta =~ /{}/ && chan =~ /{}.{}/ "\
                    "&& ondate < _{}_ && (offdate > _{}_ || "\
                    "offdate == -1)".format(sta,
                                            chan[:2],
                                            chan[3:],
                                            time,
                                            time))
    if view.record_count != 3:
        logger.warning("get_chan3: Could not determine appropriate chan3 for "\
                        "{}:{} at {}".format(sta, chan,
                            stock.epoch2str(time, "%Y%j %H:%M:%S.%s")))
        view.free()
        return None
    chan3, ondates, offdates = [], [], []
    for record in view.iter_record():
        chan, ondate, offdate = record.getv('chan', 'ondate', 'offdate')
        chan3 += [chan]
        ondates += [ondate]
        offdates += [offdate]
    chan3 = tuple(sort_chans(chan3))
    ondate = max(ondates)
    offdate = min(offdates)
    return chan3

def get3Ctr(wf_db, sta, chan3, tstart, tend):
    """
    Input: db wfdisc pointer, station name, 3 channels, start,end
    Output: Obspy Stream with 3 waveform traces (segments)
    """
    logger = logging.getLogger('db3polar')
    st3c = Stream()
    if chan3 == None:
        return None
    for chan in chan3:
        cline = "{}:{} {} - {}".format(sta, chan,
                                        stock.epoch2str(tstart,
                                                "%D %H:%M:%S.%s"),
                                        stock.epoch2str(tend,
                                                "%D %H:%M:%S.%s"))
        logger.debug("get3Ctr: Getting data for {}".format(cline))
        with ds.trdestroying(wf_db.trloadchan(tstart, tend, sta, chan)) as tr:
            if tr.record_count == 0:
                logger.warning("get3Ctr: Could not load data for {}".format(cline))
                return None
            #tr.trfilter(pfile['filter'])
            tr.record = 0
            try:
                time, samprate = tr.getv('time', 'samprate')
            except DbgetvError:
                logger.warning("get3Ctr: Could not get value 'samprate' for {}".format(cline))
                return None
            data = []
            for segment in tr.iter_record():
                tmp_data = list(segment.trdata())
                data += list(segment.trdata())
            tr.trfree()
        data = array(data)
        stats = {'station': sta, 'channel': chan, 'sampling_rate': samprate}
        stats['starttime'] = UTCDateTime(time)
        otr = Trace(data=data, header=stats)
        st3c += otr
    ns = len(st3c[0].data)
    # TODO: write subroutine to check ns, if close (w/i a few samples) try again
    if len(st3c[1].data) != ns or len(st3c[2].data) != ns:
        logger.warning('get3Ctr: {0}'.format(cline))
        logger.warning('get3Ctr: Length of data arrays not equal {0} {1} {2}'.format(len(st3c[0].data),len(st3c[1].data),len(st3c[2].data)))
        return None
    return st3c

# --------------
def calc_first_motion(ztr, ptime, tlag=0.1):
    """
    Determine the first motion of the signal 
    This is taken by using the sign of the 
    mean gradient of the waveform
    over tlag seconds After the Parrival.

    :param ztr: Z comp trace, type Trace
    :param ptime: arrival time, type UTCDateTime
    :return: fm. Returns c or d
    """
    tr = ztr.copy()
    tr2 = tr.slice(ptime, ptime+tlag)
    gg = np.gradient(tr2.data, tr2.stats.delta)
    if (np.mean(gg) < 0):
        fm = 'd'
    elif (np.mean(gg) > 0):
        fm = 'c'
    else:
        fm = '-'
    return fm

def rms(y):
    return np.sqrt(np.mean(y**2))

def calc_snr(tr, atime, tlead=1.0, tlag=1.0):
    """
    Calculate signal-to-noise ratio

    :param tr: input data, type Trace
    :param atime: reference time, type UTCDateTime
    :param tlead: seconds before atime, type float
    :param tlag: seconds after atime, type float
    :return: snr. type float
    """
    tr2 = tr.copy()
    noi = tr2.slice(atime-tlead, atime).data
    sig = tr2.slice(atime, atime+tlag).data
    sta = rms(sig)
    lta = rms(noi)
    snr = sta/lta
    #noisesq = ( lta * lta * tlead - sta * sta * tlag ) / ( tlead - tlag )
    #if ( noisesq <= 0.0 ):
    #    noisesq = 1.e-20
    #snr = sta / np.sqrt(noisesq)
    return snr
# --------------
def remove_mean(tr):
    tr.data=tr.data-np.mean(tr.data)
    return True

def cross(tr1, tr2):
    """ returns cross product of tr1 and tr2 at start for len samples """
    a = 0
    for i in range(0, tr1.stats.npts):
        a += tr1.data[i]*tr2.data[i]
    return a

def baz(z, n, e, zt, pre, post):
    """ compute back azimuth , incident angle and via Roberts et al. 1989 Journal of Geophysics """
    remove_mean(z)
    remove_mean(n)
    remove_mean(e)
    z.trim(zt-timedelta(seconds=pre), zt+timedelta(seconds=post))
    n.trim(zt-timedelta(seconds=pre), zt+timedelta(seconds=post))
    e.trim(zt-timedelta(seconds=pre), zt+timedelta(seconds=post))
    zz = cross(z, z)
    xx = cross(e, e)
    yy = cross(n, n)
    xy = cross(n, e)
    xz = cross(e, z)
    yz = cross(n, z)
    #xy = cross(n, e)
    #xz = cross(n, z)
    #yz = cross(e, z)
    power = (xx + yy + zz)/(z.stats.npts-1)
    rmsamp = math.sqrt(power)
    azi = math.atan2(-yz,-xz)
    azimuth = azi*52.29578
    if azimuth < 0:
        azimuth += 360
    z_r = zz / math.sqrt(xz*xz + yz*yz)
    a = -1 * z_r * math.cos(azi)
    b = -1 * z_r * math.sin(azi)
    error = 0.0
    cc = 0.0
    for i in range(0, z.stats.npts):
        cc = z.data[i] -a*n.data[i] -b*e.data[i]
        error += cc*cc
    coherence = 1.0 - error/zz
    incident_angle = math.atan(1.0/z_r) * 52.29578
    return azimuth, incident_angle, coherence

def velocity_observed(svel, incident_angle):
    """ return apparent velocity """
    return svel/math.sin(0.5*incident_angle/52.29578)


# --------------
def plot_polar(st,polout):

    method=1
    if(method == 0):
        labels = 'azimuth incidence azimuth_error incidence_error'.split()
    if method == 1:
        labels = 'azimuth incidence rectilinearity planarity'.split()
    if method == 2:
        labels = 'azimuth incidence rectilinearity planarity ellip'.split()
    labelnum = len(labels)
    labelnum += 3

    fig = plt.figure()
    ax = fig.add_subplot(labelnum,1,1)
    ax.plot(st[0].times(),st[0].data)
    ax = fig.add_subplot(labelnum,1,2)
    ax.plot(st[1].times(),st[1].data)
    ax = fig.add_subplot(labelnum,1,3)
    ax.plot(st[2].times(),st[2].data)

    t = polout['timestamp']/(24.*3600) + 719162
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(labelnum,1,i+4)
        ax.plot(t, polout[labels[i]])
        ax.set_xlim(t[0],t[-1])
        ax.xaxis_date()

    #t = polout['timestamp']-polout['timestamp'][0]
    #t = polout['timestamp']
    """
    ax1 = fig.add_subplot(labelnum,1,1)
    ax1.plot(t, polout['azimuth'])
    ax1.set_xlim(t[0],t[-1])
    ax1.xaxis_date()

    ax2 = fig.add_subplot(412)
    ax2.plot(t, polout['incidence'])
    ax2.set_xlim(t[0],t[-1])
    ax2.xaxis_date()

    ax3 = fig.add_subplot(413)
    ax3.plot(t, polout['rectilinearity'])
    ax3.set_xlim(t[0],t[-1])
    ax3.xaxis_date()

    ax4 = fig.add_subplot(414)
    ax4.plot(t, polout['planarity'])
    ax4.set_xlim(t[0],t[-1])
    ax4.xaxis_date()
    """
    fig.autofmt_xdate()
    plt.show()


def write_arrival_assoc_pool(dbout_arrival, dbout_assoc, 
                            evlist):
    logger = logging.getLogger('db3polar')
    dboarr = dbout_arrival
    dboass = dbout_assoc
    for i in range(len(evlist)):
        cline2 = 'STA {0} CHAN {1} ORID {2} SEAZ {3} ESAZ {4} '.format(\
                    evlist[i]['sta'],evlist[i]['chan'],evlist[i]['orid'],\
                    evlist[i]['seaz'],evlist[i]['esaz'])
        logger.debug(cline2)
        arid = dboarr.nextid('arid')
        dboarr.addv(('sta', evlist[i]['sta']),
                ('time', evlist[i]['ptime']),
                ('chan', evlist[i]['chan']),
                ('arid', arid),
                ('iphase', evlist[i]['iphase']),
                ('snr', evlist[i]['snr']),
                ('fm', evlist[i]['fm']),
                ('auth', 'db3polar')
                )
        dboass.addv(('sta', evlist[i]['sta']),
                ('arid', arid),
                ('orid', evlist[i]['orid']),
                ('phase', evlist[i]['phase']),
                ('delta', evlist[i]['delta']),
                ('seaz', evlist[i]['seaz']),
                ('esaz', evlist[i]['esaz'])
                )

def db2param(dbeos, record_number):
    """
    dbeos = event-origin join, subset optionally
    In progress work to allow multiprocessing of db loops
    """
    dbeos.record = record_number
    (orid,lat,lon,dep,otime) = dbeos.getv('orid','lat','lon','depth','time')
    print('{0} ORIGIN INFO: orid {1} lat {2}'.format(i, orid, lat))
    dbeoss = dbeos.subset('orid=={0}'.format(orid))
    print('{0} records after orid {1} subset of event-origin'.format(dbeoss.record_count, orid))


if __name__ == "__main__":
    args,parser = getargs()

    logger = setup_logger(name='db3polar', loglevel=args.logLevel,
                            logfile=args.logfile)
    dryrun = args.dryrun
    nproc = args.nproc
    dbname = args.dbname
    outdb = args.odbname

    db = ds.dbopen(dbname, 'r')
    dbwf = db.lookup(table='wfdisc')
    dbsc = db.lookup(table='sitechan')
    dbo = db.lookup(table='origin')
    dbe = db.lookup(table='event')
    dbeo = dbe.join(dbo, pattern1='prefor',pattern2='orid')
    
    if args.osub:
        osub = args.osub
        dbeos = dbeo.subset(osub)
    else:
        osub = ''
        dbeos = dbeo
    
    logger.info('{0} records in event-origin join, '.format(dbeos.record_count))
    logger.info(' with subset {0}'.format(osub))
    
    dbout = ds.dbopen(outdb, 'r+')
    dboarr = dbout.lookup(table='arrival')
    dboass = dbout.lookup(table='assoc')
    dboor = dbout.lookup(table='origin')
    dboev = dbout.lookup(table='event')
    # ---------------
    # Implement assoc-arrival join, 
    # then subset here from command line asub
    # then use the view in second loop below
    # ---------------

    dopolar = args.dopolar
    dofm = args.dofm
    dosnr = args.dosnr
    asub = "sta=~/.*/"
    if args.asub:
        asub = args.asub
        logger.info('Arrival-assoc subset {}'.format(asub))
    pretime = 5
    posttime = 5
    arrlist=[]
    assoclist=[]
    evlist=[]
    param_list=[]
    for i in range(dbeos.record_count):
        dbeos.record = i
        (orid,lat,lon,dep,otime) = dbeos.getv('orid','lat','lon','depth','time')
        logger.info('ORIGIN {0} INFO: orid {1} ({2}, {3}) {4}'.format(i, orid, lat, lon, stock.strydtime(otime)))
        dbeoss = dbeos.subset('orid=={0}'.format(orid))
        logger.debug('{0} records after orid {1} subset of event-origin'.format(dbeoss.record_count, orid))
        dbview = dbeoss.process([
                            "dbjoin assoc",
                            "dbjoin arrival"
                            ])
        #phsub = "iphase=='P'"
        #dbv = dbview.subset(phsub)
        dbv = dbview.subset(asub)
        #dbv2 = dbview.subset(phsub)
        #dbv = dbv2.subset("sta=~/EC14/")
        logger.info('{0} records in assoc-arrival join after subset {1}'.format(dbv.record_count, asub))
        tj0 = time.time()
        for j in range(dbv.record_count):
            dbv.record = j
            (atime,sta,chan,iphs,phs,dist_deg,deltim,arid) = dbv.getv('arrival.time','sta',
                                                        'chan','iphase','assoc.phase',
                                                        'delta','deltim','arid')
            # TODO: check if -999.0, calculate if needed
            (seaz,esaz) = dbv.getv('seaz','esaz')
            pick_str = '{0} {1} {2} {3} '.format(sta, chan, iphs, stock.epoch2str(atime, "%D %H:%M:%S.%s"))
            logger.debug('Input pick: {0} {1} {2} {3} '.format(sta, chan, iphs, stock.epoch2str(atime, "%D %H:%M:%S.%s")))

            chan3 = get_chan3(dbsc, sta, chan, atime)
            if chan3 is None:
                logger.warning('Could not determine chan3 for {0}, skipping.'.format(pick_str))
                continue
            tstart = atime - pretime
            tend = atime + posttime
            # For some reason, doing this subset before calling
            # get3Ctr can SOMETIMES drastically speeds things up
            if args.presub:
                wsub = 'sta=~/{0}/'.format(sta)
                dbwfs = dbwf.subset(wsub)
                st3c = get3Ctr(dbwfs, sta, chan3, tstart, tend)
            else:
                st3c = get3Ctr(dbwf, sta, chan3, tstart, tend)
            if st3c is None:
                logger.warning('st3c is None for {0}, skipping'.format(pick_str))
                continue
            # Format date and time appropriately
            st3c.detrend('demean')
            st3c.detrend('simple')
            st3c.filter('bandpass', freqmin=1.0, freqmax=10.0)

            py_date = stock.epoch2str(atime, '%Y-%m-%d', tz=None)
            ptime = stock.epoch2str(atime, '%H:%M:%S.%s', tz=None)

            tsreal = tstart
            tereal = tend
            # find the real start and end times of all components
            for i in range(3):
                tsreal = np.max([tsreal, st3c[i].stats.starttime.timestamp])
                tereal = np.min([tereal, st3c[i].stats.endtime.timestamp])
            zt = UTCDateTime(atime)
            ztr = st3c.select(component='Z')[0]
            ntr = st3c.select(component='N')[0]
            etr = st3c.select(component='E')[0]
            if dopolar:
                win_len = 1.00
                win_frac = 0.5
                polout = polarization_analysis(st3c, win_len, win_frac, 1., 10., 
                                            UTCDateTime(tsreal), UTCDateTime(tereal), 
                                            verbose=True, 
                                            method="vidale")
                #plot_polar(st3c, polout)
                # ---- Calculate polarization based on Roberts et al.
                pre = 1.0
                post = 1.0
                azi, inc, coh = baz(ztr.copy(), ntr.copy(), etr.copy(), zt, pre, post)
                logger.debug('azi {0} inc {1} coh {2}'.format(azi,inc,coh))
                logger.debug('seaz {0} esaz {1} '.format(seaz,esaz))
            # ---- Calculate polarity of PA
            fm = '-'
            if dofm:
                fm = calc_first_motion(ztr, UTCDateTime(atime), tlag=0.05)
                logger.debug('FIRST MOTION: {0} '.format(fm))
            # css3.0 requires 2 characters
            if (fm == '-'):
                logger.debug('Not doing anything')
            else:
                fm += '.'
            # - SNR
            snr = -1.00
            if dosnr:
                snr = calc_snr(ztr, UTCDateTime(atime), tlead=5., tlag=2.)
                logger.debug('SNR: {0}'.format(snr))
                #ztr.plot()
            # ----
            # Copy P arrival time from input db
            ptime_e = atime
            ev = {'sta': sta, 'chan': chan,
                    'ptime': ptime_e,
                    'iphase': iphs,
                    'phase': phs,
                    'orid': orid, 'delta': dist_deg,
                    'fm': fm, 'snr': snr,
                    'seaz': seaz, 'esaz': esaz
                }
            evlist.append(ev)
        tj1 = time.time()
        logger.info('Done with station loop for orid {0} in {1:.1f} seconds'.format(orid, tj1-tj0))

    if dryrun:
        logger.info('Finished with dryrun before writing to db, exiting.')
        exit()
    logger.info('Running pool ....')
    #pool = Pool(processes=nproc)
    #t0 = time.time()
    #result_list =  pool.map(kurtosis_repicker_keepP_jcs.kur_run_object, param_list)
    #pool.close()
    #pool.join()

    logger.info('Writing results to output db')
    write_arrival_assoc_pool(dboarr, dboass, evlist)
    logger.info('Finished processing.')
    
