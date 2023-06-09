from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report

lr_y_pred = lr_tuned.predict(x_test)

best_acc = accuracy_score(lr_y_pred, y_test) * 100
best_acc

"""After applying the Hyperparameter tuning parameters the **Accuracy** becomes **99.26686217008798 %**."""

np.sqrt(mean_squared_error(y_test, lr_y_pred))

"""Our RMS error is also very very less - 0.21661214442955293

#### **Performance Metrices**
"""

print(classification_report(y_test, lr_y_pred))

cm = confusion_matrix(y_test, lr_y_pred)

plt.figure(figsize = (12, 10))
sns.heatmap(cm, annot = True, cbar = True, cmap = "summer", linewidths='1', linecolor = 'red')

