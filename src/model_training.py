import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df=pd.read_csv("data/encoded_data.csv")

y=df["Machine failure"]

x=df.drop("Machine failure",axis=1)
print(x.shape)
print(y.shape)

# Train Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

print(x_train.shape)
print(x_test.shape)

# scalling 
scaler =StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print("scaling done")

# Logistic Regression Model
model=LogisticRegression()
model.fit(x_train,y_train)

# Prediction
y_pred=model.predict(x_test)
print(y_pred)

# Accuracy
acc=accuracy_score(y_test,y_pred)
print("Logistic Regression Accuracy",acc)

# Confusion Matrix
print(confusion_matrix(y_test,y_pred))

#Classification Report
print(classification_report(y_test,y_pred))

lr_acc=acc

# Decision Tree model
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train,y_train)

# prediction
dt_pred=dt_model.predict(x_test)

# Accuracy
print("Decision Tree Accuracy")
dt_acc=accuracy_score(y_test,dt_pred)
print(dt_acc)

# Confusion Matrix
print(confusion_matrix(y_test,dt_pred))

#Classification Report
print(classification_report(y_test,dt_pred))

# Random Forest Model
rf_model=RandomForestClassifier(random_state=42)
rf_model.fit(x_train,y_train)

# prediction
rf_pred=rf_model.predict(x_test)

# Accuracy
print("Random Forest Accuracy")
rf_acc=accuracy_score(y_test,rf_pred)
print(rf_acc)

# confusion matrix
print(confusion_matrix(y_test,rf_pred))

#classification report
print(classification_report(y_test,rf_pred))

# XGBoost Model
xgb_model=XGBClassifier(random_state=42, eval_metric="logloss")
xgb_model.fit(x_train,y_train)

# prediction
xgb_pred=xgb_model.predict(x_test)

# Accuracy
print("XGBoost Accuracy")
xgb_acc=accuracy_score(y_test,xgb_pred)
print(xgb_acc)

# confusion matrix
print(confusion_matrix(y_test,xgb_pred))

# classification Report
print(classification_report(y_test,xgb_pred))

# model comparison csv save
result={
    "Model":["Logistic Regression","Decision Tree","Random Forest","XGBoost"],
    "Accuracy":[round(lr_acc,4),round(dt_acc,4),round(rf_acc,4),round(xgb_acc,4)]
}

result_df=pd.DataFrame(result)
result_df.to_csv("data/model_result.csv",index=False)
print("model_result.csv saved")

# feature importance csv save (random forest se)
feature_names=x.columns
importance_df=pd.DataFrame({
    "Feature":feature_names,
    "Importance":rf_model.feature_importances_
})
importance_df=importance_df.sort_values("Importance",ascending=False)
importance_df.to_csv("data/feature_importance.csv",index=False)
print("feature_importance.csv saved")
