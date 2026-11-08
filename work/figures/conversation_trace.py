"""Selected conversation excerpts, checked against the source run before plotting."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/defencetest-figure-cache')
import sys
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from telephone_chain import rows,RUN

OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)
by={r['mid']:r for r in rows(RUN)} # validates verbatim excerpts against raw logs
selected=[
 ('m0059','A hypothesis, with a warning against editing'),
 ('m0061','A separate suggestion: a temporary check'),
 ('m0067','Both suggestions are linked to an actual edit'),
 ('m0075','The change is presented as something to adopt'),
 ('m0076','Another agent adopts the change'),
 ('m0092','The original speaker adopts a teammate’s patch'),
]
INK='#193b3b';MUTED='#53666b';TEAL='#176a65';RUST='#ad4827'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7.2,5.95),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,7.2),ylim=(0,5.95));ax.axis('off')
ax.text(.15,5.73,'How a tentative explanation became an accepted edit',fontsize=13,weight='bold',color=INK)
ax.text(.15,5.48,'Selected public messages · one separate-copy xarray pilot · soft guidance',fontsize=8.3,color=MUTED)
for i,(mid,label) in enumerate(selected):
 r=by[mid];top=5.22-i*.79; col=TEAL if i<2 else RUST
 ax.add_patch(FancyBboxPatch((1.12,top-.66),5.91,.66,boxstyle='round,pad=.025,rounding_size=.055',
                            facecolor='#f1f6f4' if i<2 else '#fbf1eb',edgecolor='none'))
 ax.text(.16,top-.14,r['actor'].upper(),fontsize=10.5,weight='bold',color=col)
 ax.text(.16,top-.34,f'action {r["t"]}',fontsize=8.3,color=MUTED)
 ax.text(.16,top-.51,mid,fontsize=7.3,color=MUTED)
 ax.text(1.26,top-.12,label,fontsize=8.5,weight='bold',color=col)
 lines=[]
 for phrase in r['phrases']:
  cut='…' if phrase.count('(')>phrase.count(')') else ''  # mark a quote that stops mid-parenthesis
  lines.extend(textwrap.wrap('“'+phrase+cut+'”',width=80))
 assert len(lines)<=3,(mid,lines)
 ax.text(1.26,top-.29,'\n'.join(lines),fontsize=8.5,color=INK,va='top',linespacing=1.25)
 if i<5:
  ax.annotate('',xy=(.55,top-.77),xytext=(.55,top-.56),
              arrowprops={'arrowstyle':'->','color':'#b4c4c0','lw':1})
ax.text(.15,.21,'Excerpts are verbatim; stage labels are interpretations. Arrows show selected message order,',fontsize=7.4,color=MUTED)
ax.text(.15,.065,'not a complete reply graph or proof that one message caused another agent’s action.',fontsize=7.4,color=MUTED)
for ext in ('png','pdf','svg'):
 fig.savefig(OUT/f'04_conversation_trace.{ext}',dpi=300,facecolor='white')
(OUT/'04_conversation_trace.json').write_text(json.dumps([{'stage':label,**by[mid]} for mid,label in selected],indent=2))
print('Created conversation trace in',OUT)
