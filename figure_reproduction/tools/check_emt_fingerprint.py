"""Regression check: 84 displayed values, seven programs, no flipped directions."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import numpy as np
from reproduce.legacy import load
from reproduce.emt_fingerprint import plot_fingerprint

m=load('emt/scripts/downstream/plot_emt_cellcompass_main_panels.py')
_,_,mech,_,_=m.load_tables()
original=mech.copy(deep=True)
fig=plot_fingerprint(mech,modules=m.MECH_MODULES,colors=m.MODULE_COLORS,
                     labels=m.MODULE_LABELS,cell_lines=m.CELL_LINES,
                     stimuli=m.STIMULI,save_all=lambda *_:None)
count=negative=0
for ax in fig.axes:
    for line in ax.lines:
        gid=line.get_gid()
        if not gid or not gid.startswith('program:'):continue
        _,condition,module,raw=gid.split(':')
        value=float(raw)
        assert module!='cell_cycle_growth'
        index=fig._fingerprint_modules.index(module)
        angle=np.pi/2-index*2*np.pi/7
        direction=np.array([np.cos(angle),np.sin(angle)])
        xy=np.column_stack([line.get_xdata(),line.get_ydata()])
        assert np.all(xy@direction>0), (condition,module,'flipped direction')
        np.testing.assert_allclose(xy@np.array([-direction[1],direction[0]]),0,atol=1e-12)
        assert value==float(mech.set_index('condition').loc[condition,'gen_delta_'+module])
        np.testing.assert_allclose(np.linalg.norm(xy[1]-xy[0]),.82*abs(value)/fig._fingerprint_scale)
        assert line.is_dashed()==(value<0)
        count+=1;negative+=value<0
assert count==84,count
assert len(fig.legends)==1
assert len(fig.legends[0].get_texts())==7
assert not any('growth' in t.get_text() or 'cell-cycle' in t.get_text() for t in fig.legends[0].get_texts())
assert any(t.get_text()=='Filled circles: increase\nOpen circles: decrease' for t in fig.texts)
assert mech.equals(original)
print(f'PASS: {count} unchanged displayed scores, {negative} negative values on the correct rays, 7 program labels. Full source table unchanged.')
