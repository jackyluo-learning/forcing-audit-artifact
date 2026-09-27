"""Audit workflow for v2, with unvalidated privacy/defense outputs deferred."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
fig, ax=plt.subplots(figsize=(9,2.35))
ax.set_xlim(0,9);ax.set_ylim(0,2.35);ax.axis('off')
def box(x,y,w,h,label,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',linewidth=1,edgecolor=color,facecolor='white'))
 ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10,color=color,linespacing=1.4)
def arrow(x,y,u,v):
 ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=12,lw=1,color='#444444'))
box(.08,1.35,1.55,.67,'Trained targets D','#397B35')
box(.08,.27,1.55,.67,'Matched controls C','#703B9F')
box(2.08,.25,2.15,1.8,'Same attack on M\n\nOptimizer and budget\nDecoding and success rule','#245988')
box(4.77,1.35,1.4,.67,'EMR(D)','#397B35')
box(4.77,.27,1.4,.67,'EMR(C)','#703B9F')
box(6.72,.25,2.17,1.8,'Report together\n\nTrained / control rates\nDifference + uncertainty','#8B580B')
for y in (1.685,.605):
 arrow(1.67,y,2.04,y);arrow(4.27,y,4.73,y);arrow(6.21,y,6.68,y)
fig.tight_layout(pad=.15)
for ext in ('pdf','png'):
 fig.savefig(os.path.join(os.path.dirname(__file__),'fig_audit_overview_v2.'+ext),bbox_inches='tight',pad_inches=.02,dpi=250)
