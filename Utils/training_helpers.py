from torch.optim.lr_scheduler import _LRScheduler
from scipy.special import logsumexp
import math as m
import numpy as np

#### Warning! This class when used takes about 0.15 seconds on average (peaks of up to 0.5s)
# Helps detects explosions in desired monitored values by reading the desired quantity
# using the desired method. For available methods, refer to the method code, add your own if nothing
# meets your specs.

# If relanching is activated, make sure you take care of creating checkpoints for your model called "checkpoint-X.pt",
# where X is the epoch count of your checkpoint, for the save_interval you provide

# configs: list ot dicts inputtedwith keys for: 
	# 'object'
	# 'ref'
	# 'method''
	# 'tolerance' 
	# 'action'

# During execution can take values for:
	# bad_val
	# 'reason'
	# 'comparison_pt'
	# 'difference'

# optimizer, scheduler, model: The optimizer, scheduler and model objects currently used in the training process
# saved_model_dir: If you want to relaunch, what is the path to your models?
# save_interval: After how many epochs do you save one checkpoint in your code?
class ExplosionDetector():
	def __init__(self, optimizer, scheduler, init_time, relaunching=False, saved_model_dir=None, save_interval=None, writer=None):        
        # Inputs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.saved_model_directory = saved_model_directory
        self.save_interval = save_interval
        self.writer = writer

        # State
        self.t = init_time
        self.configs = []

        # Fixed params
        self.read_methods = []


    # Main method called on object at the end of every epoch
    def step(self):

    	self.t += 1

    	# Reset the foll. states
    	self.violators = []

    	for c in self.configs:
    		unsafe = True
    		c['val'] = np.log(self.read_methods[config['method']](config['object']))
    		c['avg'] = logsumexp([c['avg'], c['val']])

    		if c.val > c.thresh: c.reason = f'violated threshold {c.thresh} by margin val-threshold: {c.val-v.threshold}'
    		elif c.val > c.avg: c.reason = f'diverged from running average of {c.avg} by margin val-avg: {c.val-c.avg} under tolerance: {c.tolerance}'
    		else: unsafe = False

    		if (unsafe):
    			if v.log: print(f'ExplosionDetector: Warning from {c.ref} -- step {self.t}-- param value {c.val} -- cause {c.reason}')
	    		if v.writer: writer.add_scalar(v.ref, v.val, self.t)
	    		if v.relaunch: need_relaunch = True
    	return


    def add(self, object_=None, ref=None, method_name=None, threshold=10**9):
    	config = Config(object_, ref, method_name, threshold)
    	self.configs.append(config)
    	return


class Config():
	def __init__(object_to_monitor, reference, read_method, mode='avg', tolerance=10, threshold=10**9, alert='log writer relaunch'):
		self.obj = object_to_monitor
		self.ref = reference
		self.read = read_method
		self.mode = mode
		self.tolerance = tolerance
		self.threshold = 10**9
		self.alerts = alert.split(' ')








