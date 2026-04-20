from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fall", "ADL"],
    yticklabels=["Fall", "ADL"]
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión (Fall = clase positiva)")
plt.show()
