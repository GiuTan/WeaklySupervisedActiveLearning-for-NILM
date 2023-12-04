import os
import logging
from datetime import datetime
import platform


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('\\')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def allocate_experiment_id_alt(log_dir):
    return datetime.now().strftime('%Y%m%d_%H%M%S-{}-{}').format(platform.node(), os.getpid())


def create_logging(log_dir, filemode):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    exp_id = allocate_experiment_id_alt(log_dir)

    log_path = os.path.join(log_dir, "%s.log" % exp_id)
    logging.basicConfig(level=logging.INFO,  # DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging
