import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sns

##Read real results
labeled_data = pd.read_csv("formatted-data.csv")
real_results = np.array(labeled_data["label"])
test_results = []
##Read test results
# with open("output.txt", "r") as results:
with open("facenet_model_output.txt", "r") as results:
    for line in results:
        if(line == "True\n"):
            test_results.append(1)
        else:
            test_results.append(0)

print(metrics.f1_score(real_results, test_results))
print(metrics.accuracy_score(real_results, test_results))
print(metrics.confusion_matrix(real_results, test_results))
### calculate metrics
accuracy = metrics.accuracy_score(real_results, test_results)
f1 = metrics.f1_score(real_results, test_results)
mcc = metrics.matthews_corrcoef(real_results, test_results)
precision = metrics.precision_score(real_results, test_results)
recall = metrics.recall_score(real_results, test_results)
conf_ma = metrics.confusion_matrix(real_results, test_results)



### show scorers and confusion matrix as a graph
fig = plt.figure()

gs = fig.add_gridspec(2, 5)

ax3 = fig.add_subplot(gs[0, 0::])

ax4 = fig.add_subplot(gs[1, 0::])



x = np.arange(5)

ax3.bar(x, [accuracy, f1, mcc, precision, recall])

ax3.set_title('Metrics')

ax3.set_xticklabels(['', 'Acc', 'F1', 'MCC', 'Precision', 'Recall'])


ax4.set_title('Confusion Matrix')

sns.set(font_scale=1.0)

ax4 = sns.heatmap( conf_ma, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt="d")

labels = ["0", "1"]

ax4.set_xticklabels(labels)

ax4.set_yticklabels(labels)

ax4.set(ylabel="True Label", xlabel="Predicted Label")



plt.show()

table = PrettyTable(['Accuracy', 'F1', 'MCC', 'Precision', 'Recall'])

table.add_row([accuracy, f1, mcc, precision, recall])

print(table)

print("")
