import logging
import sys

logger = logging.getLogger('SIGIR_experiments')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
