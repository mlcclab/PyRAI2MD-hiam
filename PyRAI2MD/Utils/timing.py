######################################################
#
# PyRAI2MD 2 module for utility tools - timing
#
# Author Jingbai Li
# Sep 8 2021
#
######################################################

import datetime

def what_is_time():
    ## This function return current time

    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def how_long(start, end):
    ## This function calculate time between start and end

    walltime = end-start
    walltime = '%5d days %5d hours %5d minutes %5d seconds' % (
        int(walltime / 86400),
        int((walltime % 86400) / 3600),
        int(((walltime % 86400) % 3600) / 60),
        int(((walltime % 86400) % 3600) % 60))
    return walltime


timing = 'Neural Networks End:  Y-M-D H:M:S Total:     0 days     0 hours     0 minutes    0 seconds'

def readtime(timing):
    timing = timing.split()
    d = timing[6]
    h = timing[8]
    m = timing[10]
    s = timing[12]
    elapsed = int(d * 86400 + h * 3600 + m * 60 + s)
    walltime = '%5d days %5d hours %5d minutes %5d seconds' % (d, h, m, s)

    return elapsed, walltime
