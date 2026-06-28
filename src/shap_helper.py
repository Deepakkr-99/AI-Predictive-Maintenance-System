import pandas as pd
import numpy as np
import shap


def get_shap_values(model, input_scaled, feature_names):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    # new shap version - shape (1, features, classes)
    if isinstance(shap_values, list):
        values = np.array(shap_values[1]).flatten()
    elif len(np.array(shap_values).shape) == 3:
        values = shap_values[0, :, 1]
    else:
        values = np.array(shap_values).flatten()

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": values
    })

    shap_df["Abs Value"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Abs Value", ascending=False)

    return shap_df.head(5)
