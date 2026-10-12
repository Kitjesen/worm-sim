import numpy as np
np.concat = np.concatenate
import runpy, sys
sys.argv = ['full_robot.py'] + sys.argv[1:]
runpy.run_path('full_robot.py', run_name='__main__')
