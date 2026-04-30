from plot_fig5_einzeln_diff import diff
from plot_fig5diff import diff1

import matplotlib.pyplot as plt

plt.plot(list(range(len(diff))),diff, label = 'a good instance', linewidth=3)
plt.plot(list(range(len(diff1))),diff1, label = 'average', linewidth=3)


plt.xlabel('iterations', fontsize=20)
plt.ylabel('loss', fontsize=20)
    #plt.ylim()
    #plt.xlim(left=70, right=140)
plt.legend(fontsize=20)

plt.tight_layout()
plt.savefig('tikzpicture_plots/fig_diff.pdf', format="pdf", bbox_inches="tight")
plt.show()