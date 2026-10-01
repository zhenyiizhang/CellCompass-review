"""Render and archive the corrected Figure 3e without rebuilding other figures."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
from reproduce.legacy import load
from reproduce.emt_fingerprint import plot_fingerprint

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/figure3e_corrected')
    out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=True)
    m=load('emt/scripts/downstream/plot_emt_cellcompass_main_panels.py')
    m.setup_style();_,_,mech,_,_=m.load_tables()
    def save(fig,_):
        for ext in ('pdf','png','svg'):
            fig.savefig(out/f'Figure3e_program_fingerprints_corrected.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    fig=plot_fingerprint(mech,modules=m.MECH_MODULES,colors=m.MODULE_COLORS,
                         labels=m.MODULE_LABELS,cell_lines=m.CELL_LINES,
                         stimuli=m.STIMULI,save_all=save)
    mech[['condition']+['gen_delta_'+v for v in fig._fingerprint_modules]].to_csv(out/'plotted_scores.csv',index=False)
    shutil.copy2(ROOT/'reproduce/emt_fingerprint.py',out/'emt_fingerprint.py')
    source=ROOT/'source/emt/downstream/emt_mechanisms/mechanism_condition_summary.csv'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'plot_settings.json').write_text(json.dumps(dict(modules=fig._fingerprint_modules,omitted_from_this_panel=['cell_cycle_growth'],colors=m.MODULE_COLORS,
        labels=m.MODULE_LABELS,cell_lines=m.CELL_LINES,stimuli=m.STIMULI,
        common_absolute_scale=fig._fingerprint_scale,source_sha256=sha),indent=2)+'\n')
    (out/'PROVENANCE.md').write_text(f'''# Figure 3e correction — 19 September 2026

The panel compares endpoint RNA-program changes across the 12 EMT conditions.
No model was fitted and no program score was changed.

## Source paths

- Final input table: `{source}`
- Renderer: `reproduce/emt_fingerprint.py` in the figure repository.
- Original plotting snapshot: `source/emt/scripts/downstream/plot_emt_cellcompass_main_panels.py`.
- Included data: `plotted_scores.csv` (12 conditions × 7 displayed programs).
- Included settings: `plot_settings.json`.

## Change

Seven pathway/state programs are displayed at the author's request. The
cell-cycle RNA score is omitted from this panel only and remains in the source
data and other analyses. Increase and decrease are explained in a compact note
rather than counted as program categories. The seven directions are evenly spaced.
Negative values no longer reverse direction onto another program's axis.
Both signs use the same magnitude scale ({fig._fingerprint_scale:.8g}), and all
negative values are shown rather than suppressing changes above −0.08.

## Caption wording

Predicted changes in seven pathway and state programs across cell lines and
stimuli: TGF-β/SMAD, EMT transcription factors, ECM remodelling, partial/basal
state, TNF/NFκB, EGFR/AP-1 and epithelial junction. Spoke length
indicates the absolute change in the corresponding RNA-program score from day 0 to day 7
on a common scale. Filled circles and solid lines indicate increases, and open
circles and dashed lines indicate decreases. Each program retains the same
direction in all conditions. The inner ring denotes zero change and the outer
ring denotes an absolute score change of {fig._fingerprint_scale:.2f}.

## Rebuild

From the cloned figure repository, with `requirements-reproduce.txt` installed:

```bash
python tools/rebuild_figure3e.py
python tools/check_emt_fingerprint.py
```

## SHA-256

Source table: `{sha}`.

The Word manuscript and Illustrator master have not been overwritten. Replace
panel 3e with the supplied vector PDF or editable SVG.
''')
    print(out)

if __name__=='__main__':main()
