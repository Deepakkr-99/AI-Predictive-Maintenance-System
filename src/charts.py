import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Sabhi charts me light theme layout styling apply karne ka helper
def apply_simple_layout(fig, height=300):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111827", family="Inter", size=12), # Dark text label for light theme
        margin=dict(t=40, b=20, l=10, r=10),
        uirevision="constant" # Keeps the zoom/pan/legend state intact during Streamlit reruns
    )
    # Light theme grid lines and axes borders
    fig.update_xaxes(
        gridcolor="#e5e7eb",
        tickfont=dict(color="#374151"),
        linecolor="#cbd5e1"
    )
    fig.update_yaxes(
        gridcolor="#e5e7eb",
        tickfont=dict(color="#374151"),
        linecolor="#cbd5e1"
    )
    return fig

# Status probability bar chart (Teal + Red)
def make_prob_chart(working, failure):
    df = pd.DataFrame({
        "Status": ["Working", "Failure"],
        "Probability": [working, failure]
    })
    
    # Working ke liye Teal aur Failure ke liye Red color mapping
    fig = px.bar(
        df, x="Status", y="Probability", color="Status", text="Probability",
        color_discrete_map={"Working": "#0f766e", "Failure": "#dc2626"}
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return apply_simple_layout(fig, 250)

# Models comparison accuracies bar chart (Teal gradient scale)
def make_model_chart(df):
    fig = px.bar(
        df, x="Model", y="Accuracy", color="Accuracy", text="Accuracy",
        color_continuous_scale=["#99f6e4", "#14b8a6", "#0f766e"]
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(coloraxis_showscale=False)
    return apply_simple_layout(fig, 300)

# Feature importance horizontal bar chart (Teal gradient scale)
def make_feature_chart(df):
    df_sorted = df.sort_values("Importance")
    fig = px.bar(
        df_sorted, x="Importance", y="Feature", orientation="h", color="Importance",
        color_continuous_scale=["#99f6e4", "#14b8a6", "#0f766e"]
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_simple_layout(fig, 320)

# Confusion matrix heatmap (YlGnBu colorful scale)
def make_confusion_chart(matrix):
    labels = ["Normal", "Failure"]
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        x=labels,
        y=labels
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_simple_layout(fig, 300)

# ROC curve trace line chart
def make_roc_chart(fpr, tpr, auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC",
                             line=dict(color="#0f766e", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#94a3b8", dash="dash")))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title=f"AUC = {auc_score}"
    )
    return apply_simple_layout(fig, 300)

# Local SHAP feature values chart (Red for risk, Teal for normal)
def make_shap_chart(shap_df):
    shap_sorted = shap_df.sort_values("SHAP Value")
    colors = ["#dc2626" if val > 0 else "#0f766e" for val in shap_sorted["SHAP Value"]]
    
    fig = go.Figure(go.Bar(
        x=shap_sorted["SHAP Value"],
        y=shap_sorted["Feature"],
        orientation="h",
        marker_color=colors
    ))
    fig.update_layout(xaxis_title="SHAP Value")
    return apply_simple_layout(fig, 280)

# EDA Correlation Heatmap (YlGnBu colorful scale)
def make_correlation_chart(df):
    corr = df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(coloraxis_showscale=False)
    return apply_simple_layout(fig, 450)

# Failure class target distribution (Teal + Red)
def make_failure_distribution_chart(df):
    counts = df["Machine failure"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Status"] = counts["Status"].map({0: "Normal", 1: "Failure"})
    
    fig = px.bar(
        counts, x="Status", y="Count", color="Status", text="Count",
        color_discrete_map={"Normal": "#0f766e", "Failure": "#dc2626"},
        title="Distribution of Machine Failure vs Normal"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return apply_simple_layout(fig, 300)

# Scatter plot features relationship check (Teal + Red)
def make_eda_scatter_chart(df, x_col, y_col):
    df_plot = df.copy()
    df_plot["Status"] = df_plot["Machine failure"].map({0: "Normal", 1: "Failure"})
    fig = px.scatter(
        df_plot, x=x_col, y=y_col, color="Status",
        color_discrete_map={"Normal": "#0f766e", "Failure": "#dc2626"},
        title=f"{x_col} vs {y_col}"
    )
    return apply_simple_layout(fig, 350)
