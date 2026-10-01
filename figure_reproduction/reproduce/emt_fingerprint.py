"""Signed EMT fingerprints with fixed program directions (Figure 3e).

Only the visual encoding is revised. Input endpoint scores are not modified.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def plot_fingerprint(mech, *, modules, colors, labels, cell_lines, stimuli, save_all):
    # Figure 3e focuses on pathway/state programs. Keep cell-cycle scores in
    # the source data and other panels, but do not draw them in this panel.
    modules = [m for m in modules if m != 'cell_cycle_growth']
    assert len(modules) == 7 and len(set(modules)) == 7
    display = dict(labels)
    display.update(TGFB_SMAD_targets='TGF-β/SMAD', TNF_NFKB_inflammatory='TNF/NFκB')
    values = mech[[f'gen_delta_{m}' for m in modules]].to_numpy(float)
    assert np.isfinite(values).all()
    scale = max(1.25, float(np.abs(values).max()))
    angles = np.pi / 2 - np.arange(len(modules)) * 2 * np.pi / len(modules)
    indexed = mech.set_index('condition')
    with plt.rc_context({'font.family':'Arial', 'font.size':9, 'text.color':'black',
                         'pdf.fonttype':42, 'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(10.8, 6.8))
        grid = fig.add_gridspec(4, 3, left=.13, right=.72, bottom=.13, top=.87,
                               wspace=.28, hspace=.15)
        fig.text(.025, .965, 'Predicted changes in gene programs (day 7 − day 0)\nacross cancer contexts',
                 fontsize=12, weight='bold', va='top', color='black')
        for row, cell in enumerate(cell_lines):
            for col, stimulus in enumerate(stimuli):
                condition = f'{cell}_{stimulus}'
                ax = fig.add_subplot(grid[row,col])
                ax.set_aspect('equal'); ax.axis('off')
                ax.set_xlim(-1.12,1.12); ax.set_ylim(-1.12,1.12)
                ax.set_label(condition)
                ax.add_patch(Circle((0,0),.22,facecolor='white',edgecolor='#BBBBBB',lw=.6))
                ax.add_patch(Circle((0,0),1.04,fill=False,edgecolor='#E5E5E5',lw=.6))
                for module,angle in zip(modules,angles):
                    value = float(indexed.loc[condition,f'gen_delta_{module}'])
                    direction = np.array([np.cos(angle),np.sin(angle)])
                    ax.plot(*np.stack([.22*direction,1.04*direction]).T,color='#E5E5E5',lw=.6,zorder=0)
                    radius = .22 + .82 * abs(value) / scale
                    start,end = .22*direction,radius*direction
                    negative = value < 0
                    line, = ax.plot([start[0],end[0]],[start[1],end[1]],
                                   color=colors[module],lw=1.8,
                                   ls=(0,(2,1.5)) if negative else '-',zorder=3)
                    line.set_gid(f'program:{condition}:{module}:{value:.17g}')
                    if value != 0:
                        ax.plot(end[0],end[1],marker='o',ms=4.4,ls='',
                                markerfacecolor='white' if negative else colors[module],
                                markeredgecolor=colors[module],markeredgewidth=.9,zorder=4)
                if row == 0:
                    ax.set_title('TGF-β1' if stimulus == 'TGFB1' else stimulus,
                                 fontsize=9,weight='bold',color='black',pad=7)
                if col == 0:
                    ax.text(-.28,.5,cell,transform=ax.transAxes,ha='right',va='center',
                            fontsize=9,color='black')
        order = ['TGFB_SMAD_targets','EMT_TF_axis','ECM_remodeling',
                 'partial_basal_hybrid','TNF_NFKB_inflammatory','EGFR_AP1_MAPK',
                 'epithelial_junction']
        handles = [Line2D([],[],color=colors[m],lw=1.8,label=display[m]) for m in order]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.74,.86),
                   frameon=False,fontsize=9,labelcolor='black',labelspacing=.8,
                   handlelength=1.8,borderaxespad=0)
        fig.text(.74,.43,'Filled circles: increase\nOpen circles: decrease',
                 fontsize=9,color='black',va='top',linespacing=1.6)
        fig._fingerprint_scale = scale
        fig._fingerprint_modules = modules
        save_all(fig,'panel04_mechanism_fingerprint_cards')
    return fig
