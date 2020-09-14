import pandas as pd
import numpy as np
from datetime import datetime
import seaborn
import matplotlib.pyplot as plt
seaborn.set_style('darkgrid')

def __to_percent1(y, position):
    y = y * 100.0
    return "{:.1f}%".format(y)

def plot_roc(target, predicted_proba, title, save_png=''):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(target, predicted_proba)
        auc_plot = roc_auc_score(target, predicted_proba)        
        plt.figure()
        plt.plot(fpr, tpr, '-', alpha=.8, color='red', lw=1.5, label= title + ' (auc = %0.3f)' % auc_plot)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')

        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(__to_percent1))
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(__to_percent1))
        plt.xlabel('Non default cases', fontsize=15)
        plt.ylabel('Default cases', fontsize=15)

        plt.title("\nROC curve - {}\n".format(title), fontsize=18)
        plt.legend(loc="lower right", fontsize=15)
        
        if save_png != '':
                plt.savefig(save_png, format="png")
        else:
                plt.show()
                
def plot_grade_roc(target, grade, predicted_proba, title, save_png=''):
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(target, predicted_proba)
        fpr_plot, tpr_plot, _ = roc_curve(target, grade)
        raw_auc_plot = roc_auc_score(target, predicted_proba)
        new_grade_auc_plot = roc_auc_score(target, grade)

        plt.figure()
        plt.plot(fpr, tpr, '-', color='grey', alpha=.3, label="Raw PD (auc = %0.3f)" % raw_auc_plot)
        plt.plot(fpr_plot, tpr_plot, 'o-', color='red', alpha=.8, lw=1.5, label= title + ' (auc = %0.3f)' % new_grade_auc_plot)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(__to_percent1))
        plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(__to_percent1))
        plt.xlabel('Non wasted policies', fontsize=15)
        plt.ylabel('Wasted policies', fontsize=15)

        plt.title("\nROC curve - {}\n".format(title), fontsize=18)
        plt.legend(loc="lower right", fontsize=15)

        bbox_props = dict(boxstyle="circle,pad=0.3", fc="white", ec="#2769a6", lw=1)
        bbox_props2 = dict(boxstyle="circle,pad=0.3", fc="white", ec="red", lw=1)
        bbox_props3 = dict(boxstyle="circle,pad=0.3", fc="white", ec="blue", lw=1)
        
        for i in range(0,6):
                if i >= 1 and i <= 6:
                        try:
                                plt.text(fpr_plot[i] - .01, tpr_plot[i] + .05, "%s" % (6 - i), color="red", ha="center", va="center", size=15, bbox=bbox_props2)
                        except:
                                pass
        
        if save_png != '':
                plt.savefig(save_png, format="png")
        plt.show()