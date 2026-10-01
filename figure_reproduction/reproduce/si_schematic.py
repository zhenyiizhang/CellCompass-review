"""Supplementary Figure 1: CellCompass model selection and development of new
dynamics, drawn as a plain monochrome flow diagram that follows the caption."""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch

from .style import apply_style,emit,BLACK,BLUE,GRAY,PLOT_TEXT_SIZE

FILL='#F4F4F4'

def box(ax,x,y,w,h,title,lines=(),*,edge=BLACK,fill='white',lw=.8,title_size=9.5,center=False):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,linewidth=lw,zorder=2))
    top=y+h-.8
    tx=x+w/2 if center else x+.9
    ax.text(tx,top,title,fontsize=title_size,fontweight='bold',ha='center' if center else 'left',va='top',color=BLACK,linespacing=1.2,zorder=3)
    if lines:
        offset=2.3+1.5*(title.count('\n'))
        ax.text(tx,top-offset,'\n'.join(lines),fontsize=PLOT_TEXT_SIZE,ha='center' if center else 'left',va='top',color=BLACK,linespacing=1.3,zorder=3)
    return (x,y,w,h)

def arrow(ax,p0,p1,*,color=BLACK,lw=.9,ls='-',shrink=1.2):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle='-|>',mutation_scale=9,linewidth=lw,color=color,linestyle=ls,
                                 shrinkA=shrink,shrinkB=shrink,zorder=4))

def right(b):x,y,w,h=b;return (x+w,y+h/2)
def left(b):x,y,w,h=b;return (x,y+h/2)
def bottom(b,f=.5):x,y,w,h=b;return (x+w*f,y)
def top(b,f=.5):x,y,w,h=b;return (x+w*f,y+h)

def model_lifecycle(stem):
    apply_style()
    fig=plt.figure(figsize=(11.6,4.9))
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,116);ax.set_ylim(9,58);ax.set_aspect('equal');ax.axis('off')

    # Row 1: study definition, model selection and the shared reporting path.
    y1,h1=39,12.5
    q=box(ax,3,y1,23,h1,'Biological question and data',
          ['biological question','time-indexed cell states','additional measurements','cellular outcome to predict'])
    f=box(ax,30,y1,26,h1,'Problem formulation',
          ['how present cells connect to the outcome','comparison used for evaluation','required dynamical quantity'])
    d=box(ax,60,y1+2.0,22,8.5,'Formulation available in\nthe dynamical engine?',(),fill=FILL,title_size=9.0,center=True)
    r=box(ax,86,y1,27,h1,'Fitting, evaluation and reporting',
          ['fit the selected model','simulate from the same initial cells','compare with observed outcomes','analyse the generated trajectories'])
    arrow(ax,right(q),left(f));arrow(ax,right(f),left(d));arrow(ax,right(d),left(r))
    ax.text((right(d)[0]+left(r)[0])/2,right(d)[1]+.9,'yes',fontsize=PLOT_TEXT_SIZE,ha='center',va='bottom')

    # Row 2: development of new dynamics.
    y2=16
    h2=12.0
    p=box(ax,3,y2,25,h2,'Proposal for new dynamics',
          ['biological improvement','variables and equations','population mass and stochasticity','fitting measurements','prediction inputs and evaluation'])
    v=box(ax,32,y2,26,h2,'Independent reviews',
          ['relation can be learned from the data','implementation matches the proposal','prediction uses only information','available at initialization'])
    t=box(ax,62,y2,22,h2,'Candidate tests',
          ['numerical feasibility','recovery of the biological relation','stability'])
    c=box(ax,88,y2,25,h2,'Final confirmation',
          ['repeats the complete comparison','stores the model and results','used for reporting'],edge=BLUE,lw=1.1)
    # "no" branch: decision -> proposal.
    yj=y2+h2+4.0
    xd=bottom(d)[0];xp=top(p)[0]
    ax.plot([xd,xd,xp],[d[1],yj,yj],color=BLACK,lw=.9,zorder=4,solid_capstyle='round')
    arrow(ax,(xp,yj),top(p),shrink=0.0)
    ax.text(xd+.9,(d[1]+yj)/2,'no',fontsize=PLOT_TEXT_SIZE,ha='left',va='center')
    arrow(ax,right(p),left(v));arrow(ax,right(v),left(t));arrow(ax,right(t),left(c))
    # Revision loop below the row.
    yl=y2-3.0
    for src in (v,t):
        ax.plot([bottom(src)[0],bottom(src)[0]],[src[1],yl],color=GRAY,lw=.8,ls=(0,(3,2)),zorder=4)
    ax.plot([xp,bottom(t)[0]],[yl,yl],color=GRAY,lw=.8,ls=(0,(3,2)),zorder=4)
    arrow(ax,(xp,yl),bottom(p),color=GRAY,lw=.8,ls=(0,(3,2)),shrink=0.0)
    ax.text((xp+bottom(t)[0])/2,yl-1.0,'failed review or test returns to the proposal',fontsize=PLOT_TEXT_SIZE,ha='center',va='top',color=BLACK)
    # Confirmed model joins the reporting path.
    xc=top(c)[0]
    arrow(ax,(xc,top(c)[1]),(xc,r[1]),color=BLUE,lw=1.0,shrink=1.0)
    ax.text(xc+1.0,(top(c)[1]+r[1])/2,'confirmed model',fontsize=PLOT_TEXT_SIZE,ha='left',va='center',color=BLACK)
    emit(fig,stem)
