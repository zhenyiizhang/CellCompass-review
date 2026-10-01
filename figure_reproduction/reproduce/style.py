"""Manuscript figure style shared by the redrawn supplementary panels.

Typography and palette follow the CellCompass/CytoBridge figure guide: Arial,
black text, 9 pt plot text, 12 pt bold group titles, 14 pt bold lower-case panel
labels, open axes without grids, black for observed/reference data, blue for the
model and red for the paired comparison.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

BLACK='#000000'
BLUE='#2166AC'
RED='#B2182B'
GRAY='#7F7F7F'
LIGHT='#BBBBBB'
GRID='#DDDDDD'

PANEL_LABEL_SIZE=14.0
GROUP_TITLE_SIZE=12.0
PLOT_TEXT_SIZE=9.0

# Small conventional palette for categorical series that are not a
# model/reference pair (route classes, perturbation modules).
TAB={'red':'#D62728','orange':'#FF7F0E','purple':'#9467BD','green':'#2CA02C','brown':'#8C564B','gray':'#7F7F7F','blue':'#1F77B4','cyan':'#17BECF'}

_ARIAL_REGISTERED=False

def register_arial():
    """Make the system Arial faces available to Matplotlib on macOS."""
    global _ARIAL_REGISTERED
    if _ARIAL_REGISTERED:return
    for path in Path('/System/Library/Fonts/Supplemental').glob('Arial*.ttf'):
        try:font_manager.fontManager.addfont(str(path))
        except Exception:pass
    _ARIAL_REGISTERED=True

def apply_style():
    register_arial()
    mpl.rcParams.update({
        'font.family':'Arial','font.sans-serif':['Arial'],
        'font.size':PLOT_TEXT_SIZE,'axes.titlesize':PLOT_TEXT_SIZE,'axes.titleweight':'normal',
        'axes.labelsize':PLOT_TEXT_SIZE,'xtick.labelsize':PLOT_TEXT_SIZE,'ytick.labelsize':PLOT_TEXT_SIZE,
        'legend.fontsize':PLOT_TEXT_SIZE,'legend.frameon':False,'legend.title_fontsize':PLOT_TEXT_SIZE,
        'axes.linewidth':0.65,'axes.edgecolor':BLACK,'xtick.major.width':0.6,'ytick.major.width':0.6,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.direction':'out','ytick.direction':'out',
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
        'lines.linewidth':1.1,'lines.markersize':4,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'text.color':BLACK,'axes.labelcolor':BLACK,'axes.titlecolor':BLACK,'xtick.color':BLACK,'ytick.color':BLACK,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
        'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular',
        'figure.dpi':100,'savefig.bbox':'tight','savefig.pad_inches':0.04,
    })

def clean(ax,*,grid=False):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    if grid:ax.grid(color=GRID,linewidth=.45,alpha=.55,zorder=0)
    else:ax.grid(False)
    ax.set_axisbelow(True)

def panel_label(ax,letter,x=-0.12,y=1.04):
    """Lower-case bold 14 pt panel label anchored to the axes corner."""
    return ax.text(x,y,letter.lower(),transform=ax.transAxes,fontsize=PANEL_LABEL_SIZE,fontweight='bold',ha='left',va='bottom',color=BLACK)

def group_title(ax,title,x=0.0,y=1.04):
    return ax.text(x,y,title,transform=ax.transAxes,fontsize=GROUP_TITLE_SIZE,fontweight='bold',ha='left',va='bottom',color=BLACK)

def legend(ax,handles=None,labels=None,**kwargs):
    kwargs.setdefault('frameon',False);kwargs.setdefault('fontsize',PLOT_TEXT_SIZE)
    kwargs.setdefault('handletextpad',.5);kwargs.setdefault('borderaxespad',.3)
    if handles is None:return ax.legend(**kwargs)
    if labels is None:return ax.legend(handles=handles,**kwargs)
    return ax.legend(handles,labels,**kwargs)

def embedding_axis(ax):
    """Boxless embedding panel with equal aspect and no ticks."""
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_aspect('equal',adjustable='box')

def emit(fig,stem,dpi=320):
    """Vector PDF plus PNG preview. build.capture() redirects the destination."""
    fig.savefig(stem+'.pdf',bbox_inches='tight')
    fig.savefig(stem+'.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
