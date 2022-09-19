from __future__ import print_function
import numpy as np
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import getdist
from getdist import plots, MCSamples, loadMCSamples
import getdist, IPython
import pylab as plt
print('GetDist Version: %s, Matplotlib version: %s'%(getdist.__version__, plt.matplotlib.__version__))
#matplotlib 2 doesn't seem to work well without usetex on
plt.rcParams['text.usetex']=True
#import seaborn as sns
#sns.set()

import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_context("talk")
#sns.set_context("poster")
palette = sns.color_palette()

#plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'lines.linewidth':3})
#plt.rcParams.update({'usetex': True})

#cp = sns.color_palette()
plt.rcParams.update({'font.size': 22})

import os # looking for files names chains,
from glob import glob
PATH = './'
file_names = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], 'chain*'))]
file_names

for idx, file in enumerate(file_names):
    inChain = np.loadtxt(file,delimiter=' ')
    nsamps, npar = inChain.shape
    outChain = np.zeros((nsamps,npar+1))
    outChain[:,1:] = np.copy(inChain)
    outChain[:,0] = 1.
    np.savetxt('./convert_{}.txt'.format(idx+1),outChain)

samples = loadMCSamples('./convert', settings={'ignore_rows': 0.3})

# 1D marginalized plot
g = plots.get_single_plotter(width_inch=6)
g.plot_1d(samples, 'p1')
plt.show()

from getdist import plots # gives us a two dimensional posterio, two dimensional posterior, doing two things, showing shading, shading is smoothed density  plot of samples. The relative density is prportional to the posterior. SHows 1 and 2 sigmas contours. Percentage is volume under the curve
g = plots.getSinglePlotter()
g.plot_2d(samples, ['p1', 'p2'],shaded=True)


g = plots.getSubplotPlotter(width_inch=8)
#g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
g.settings.axes_fontsize = 20
g.settings.lab_fontsize = 22
g.settings.legend_fontsize = 20
g.triangle_plot([samples], ['p1', 'p2'],
    filled_compare=True,
    legend_labels=['Samples'],
    legend_loc='upper right',
    line_args=[{'ls':'-', 'color':'green'}],
    contour_colors=['green'])
g.export('Triangle.pdf')

# Many other things you can do besides plot, e.g. get latex
# Default limits are 1: 68%, 2: 95%, 3: 99% probability enclosed
# See  https://getdist.readthedocs.io/en/latest/analysis_settings.html
# and examples for below for changing analysis settings
# (e.g. 2hidh limits, and how they are defined)
print(samples.getInlineLatex('p1',limit=2))
print(samples.getInlineLatex('p2',limit=2))

print(samples.getTable().tableTex())

print(samples.PCA(['p1','p2']))

stats = samples.getMargeStats()
lims0 = stats.parWithName('p1').limits
lims1 = stats.parWithName('p2').limits

for conf, lim0, lim1 in zip(samples.contours,lims0, lims1):
    print('p1 %s%% lower: %.3f upper: %.3f (%s)'%(conf, lim0.lower, lim0.upper, lim0.limitType()))
    print('p2 %s%% lower: %.3f upper: %.3f (%s)'%(conf, lim1.lower, lim1.upper, lim1.limitType()))
plt.show()