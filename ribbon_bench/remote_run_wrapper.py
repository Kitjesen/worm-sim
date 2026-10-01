"""Run the same physics core with NumPy 1.x/2.x on the remote GPU host."""
import numpy as np
import runpy
import sys
from pathlib import Path

if not hasattr(np, 'concat'):
    np.concat = np.concatenate  # Upstream uses the NumPy 2.x spelling.
sys.argv[0] = 'full_robot.py'
runpy.run_path(str(Path(__file__).with_name('full_robot.py')), run_name='__main__')
