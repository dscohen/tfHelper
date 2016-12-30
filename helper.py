import math
import time
import datetime
import sys
import tensorflow as tf

# def listify(input):
#     ti = []
#     for i in input:
#         if type(j) == list or if:
#             ti.append(j)

def get_last_timestep(tensor):
    tensor = tf.transpose(tensor,[1,0,2])
    return tf.gather(tensor,tensor.get_shape()[0]-1)

def mvg_avg(n,point,avg):
    if math.isnan(point):
        return point
    return (avg*n + point)/(n+1.)

def read_time(time):
    time_str = ''
    hr = time/(60*60)
    minute = time/60
    sec = time%60
    return map(int,[hr,minute,sec])

def printProgress (iteration, total, t,time, prefix = '', suffix = '', decimals = 1, barLength = 100,elapsed_t = None,loss=None,metric = None,eta = None,feed_dict = None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    time_in = time[0]
    time_out = time[1]
    time = (time_out - time_in) * (total-iteration)
    time = read_time(time)
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = 'X' * filledLength + '-' * (barLength - filledLength)
    loss_val        = loss.eval(feed_dict = feed_dict) if loss is not None else float('nan')
    metric_val      = metric.eval(feed_dict = feed_dict) if metric is not None else float('nan')
    t               = [loss_val,metric_val]
    # t               = [mvg_avg(iteration,loss_val,t[0]), mvg_avg(iteration,metric_val,t[1]),time]
    sys.stdout.write('\r%s |%s| %s%s %s Loss: %.2f Accuracy: %.6f ETA: %s:%s:%s' % (prefix, bar, percents, '%', '',t[0],t[1],time[0],time[1],time[2])),
    sys.stdout.flush()
    # if iteration == total:
    #     sys.stdout.write('\n')
    #     sys.stdout.flush()
    return t
