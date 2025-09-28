import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, mean_squared_error,
    r2_score, mean_absolute_error, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Student Career Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('student_career_performance.csv')

        # =====================
        # Part A – Data Preprocessing
        # =====================
        df = df.drop_duplicates()

        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

        for col in ['CGPA', 'Hours_Study', 'Sleep_Hours']:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = np.clip(df[col], lower, upper)

        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'student_career_performance.csv' is in the project directory.")
        return None

# Main app
def main():
    st.title("🎓 Student Career Performance Analytics Dashboard")
    st.markdown("---")

    df = load_data()
    if df is None:
        return

    st.sidebar.title("📊 Dashboard Controls")

    st.sidebar.subheader("📈 Data Overview")
    st.sidebar.metric("Total Students", len(df))
    st.sidebar.metric("Placed Students", df['Placed'].sum())
    st.sidebar.metric("Placement Rate", f"{df['Placed'].mean()*100:.1f}%")

    st.sidebar.subheader("🔍 Filters")
    cgpa_range = st.sidebar.slider("CGPA Range", float(df['CGPA'].min()), float(df['CGPA'].max()), (float(df['CGPA'].min()), float(df['CGPA'].max())), step=0.1)
    study_hours_range = st.sidebar.slider("Study Hours Range", float(df['Hours_Study'].min()), float(df['Hours_Study'].max()), (float(df['Hours_Study'].min()), float(df['Hours_Study'].max())), step=0.1)
    placement_filter = st.sidebar.selectbox("Placement Status", ["All", "Placed", "Not Placed"])

    filtered_df = df[(df['CGPA'] >= cgpa_range[0]) & (df['CGPA'] <= cgpa_range[1]) & (df['Hours_Study'] >= study_hours_range[0]) & (df['Hours_Study'] <= study_hours_range[1])]

    if placement_filter == "Placed":
        filtered_df = filtered_df[filtered_df['Placed'] == 1]
    elif placement_filter == "Not Placed":
        filtered_df = filtered_df[filtered_df['Placed'] == 0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Filtered Students", len(filtered_df))
    with col2:
        st.metric("Average CGPA", f"{filtered_df['CGPA'].mean():.2f}")
    with col3:
        st.metric("Avg Study Hours", f"{filtered_df['Hours_Study'].mean():.1f}")
    with col4:
        st.metric("Placement Rate", f"{filtered_df['Placed'].mean() * 100:.1f}%")

    st.markdown("---")

    model_choice = st.sidebar.multiselect("Select Models for Placement Prediction", ["Random Forest", "Logistic Regression"], default=["Random Forest"])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Overview", "📈 Correlations", "🎯 Predictions", "📋 Data Table", "🔍 Insights", "🎯 Score Prediction", "📈 Linear Regression"])

    with tab3:
        st.subheader("🎯 Placement Prediction Models")
        feature_cols = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA', 'Placement_Score']
        X = filtered_df[feature_cols]
        y = filtered_df['Placed']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

        results = {}

        if "Random Forest" in model_choice:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            y_pred_rf = rf.predict(X_test_scaled)
            y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
            results["Random Forest"] = {
                "accuracy": accuracy_score(y_test, y_pred_rf),
                "precision": precision_score(y_test, y_pred_rf),
                "recall": recall_score(y_test, y_pred_rf),
                "f1": f1_score(y_test, y_pred_rf)
            }
            st.write("### 🌲 Random Forest Results")
            st.json(results["Random Forest"])

            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig_cm_rf = px.imshow(cm_rf, text_auto=True, title="Random Forest Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm_rf, use_container_width=True)

            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
            roc_auc_rf = auc(fpr_rf, tpr_rf)
            fig_roc_rf = go.Figure()
            fig_roc_rf.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f"RF AUC = {roc_auc_rf:.2f}"))
            fig_roc_rf.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash="dash"), name="Random"))
            fig_roc_rf.update_layout(title="Random Forest ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc_rf, use_container_width=True)

        if "Logistic Regression" in model_choice:
            log_reg = LogisticRegression(max_iter=1000)
            log_reg.fit(X_train_scaled, y_train)
            y_pred_log = log_reg.predict(X_test_scaled)
            y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]
            results["Logistic Regression"] = {
                "accuracy": accuracy_score(y_test, y_pred_log),
                "precision": precision_score(y_test, y_pred_log),
                "recall": recall_score(y_test, y_pred_log),
                "f1": f1_score(y_test, y_pred_log)
            }
            st.write("### 📈 Logistic Regression Results")
            st.json(results["Logistic Regression"])

            cm_log = confusion_matrix(y_test, y_pred_log)
            fig_cm_log = px.imshow(cm_log, text_auto=True, title="Logistic Regression Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm_log, use_container_width=True)

            fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
            roc_auc_log = auc(fpr_log, tpr_log)
            fig_roc_log = go.Figure()
            fig_roc_log.add_trace(go.Scatter(x=fpr_log, y=tpr_log, mode='lines', name=f"LogReg AUC = {roc_auc_log:.2f}"))
            fig_roc_log.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash="dash"), name="Random"))
            fig_roc_log.update_layout(title="Logistic Regression ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc_log, use_container_width=True)

        if len(results) > 1:
            st.subheader("📊 Model Comparison")
            comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
            fig_comp = px.bar(comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                              x="Model", y="Score", color="Metric", barmode="group",
                              title="Comparison of Model Performance Metrics")
            st.plotly_chart(fig_comp, use_container_width=True)

    # Part B - Linear Regression for Placement Score Prediction
    with tab7:
        st.subheader("📈 Linear Regression - Placement Score Prediction")
        
        # Prepare data for linear regression
        feature_cols_reg = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA']
        X_reg = filtered_df[feature_cols_reg]
        y_reg = filtered_df['Placement_Score']
        
        # Split data
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        # Scale features
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)
        
        # Train Linear Regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train_reg_scaled, y_train_reg)
        
        # Make predictions
        y_pred_reg = lr_model.predict(X_test_reg_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            st.metric("MSE", f"{mse:.2f}")
        
        # Predicted vs Actual plot
        fig_pred_actual = go.Figure()
        fig_pred_actual.add_trace(go.Scatter(
            x=y_test_reg, y=y_pred_reg, mode='markers', name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        fig_pred_actual.add_trace(go.Scatter(
            x=[y_test_reg.min(), y_test_reg.max()], y=[y_test_reg.min(), y_test_reg.max()],
            mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')
        ))
        fig_pred_actual.update_layout(
            title="Linear Regression: Predicted vs Actual Placement Scores",
            xaxis_title="Actual Placement Score", yaxis_title="Predicted Placement Score"
        )
        st.plotly_chart(fig_pred_actual, use_container_width=True)
        
        # Residuals plot
        residuals = y_test_reg - y_pred_reg
        fig_resid = px.scatter(x=y_pred_reg, y=residuals, title="Residuals vs Predicted Values",
                              labels={'x': 'Predicted Values', 'y': 'Residuals'})
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, use_container_width=True)
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'Feature': feature_cols_reg,
            'Coefficient': lr_model.coef_
        }).sort_values('Coefficient', key=abs, ascending=True)
        
        fig_coef = px.bar(feature_importance, x='Coefficient', y='Feature', orientation='h',
                         title="Linear Regression Feature Coefficients")
        st.plotly_chart(fig_coef, use_container_width=True)
        
        # Interactive prediction
        st.subheader("🔮 Predict Placement Score for New Student")
        col1, col2, col3 = st.columns(3)
        with col1:
            hours_study = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=6.0, key="lr_hours")
            sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=8.0, key="lr_sleep")
        with col2:
            internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1, key="lr_internships")
            projects = st.number_input("Number of Projects", min_value=0, max_value=20, value=2, key="lr_projects")
        with col3:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, key="lr_cgpa")
        
        if st.button("Predict Placement Score", key="lr_predict"):
            new_student = np.array([[hours_study, sleep_hours, internships, projects, cgpa]])
            new_student_scaled = scaler_reg.transform(new_student)
            predicted_score = lr_model.predict(new_student_scaled)[0]
            
            st.success(f"🎯 **Predicted Placement Score: {predicted_score:.1f}**")
            
            # Performance interpretation
            if predicted_score >= 90:
                st.success("🌟 Excellent! This student is likely to get a very high placement score.")
            elif predicted_score >= 80:
                st.info("👍 Good! This student should perform well in placements.")
            elif predicted_score >= 70:
                st.warning("⚠️ Average performance expected. Consider additional preparation.")
            else:
                st.error("📚 Below average. Significant improvement needed for better placement prospects.")

if __name__ == "__main__":
    main()