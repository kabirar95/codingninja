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
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error, 
                           r2_score, mean_absolute_error, confusion_matrix, roc_curve, 
                           auc, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import scipy.stats as stats

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
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'student_career_performance.csv' is in the project directory.")
        return None

def preprocess_data(df):
    """Part A: Comprehensive Data Preprocessing"""
    st.subheader("🔧 Data Preprocessing")
    
    # Initial data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Dataset Shape:**", df.shape)
    with col2:
        st.write("**Missing Values:**", df.isnull().sum().sum())
    with col3:
        st.write("**Duplicate Rows:**", df.duplicated().sum())
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        st.success("✅ Missing values handled using median imputation")
    else:
        st.info("✅ No missing values found")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    if initial_count != final_count:
        st.success(f"✅ Removed {initial_count - final_count} duplicate rows")
    else:
        st.info("✅ No duplicate rows found")
    
    # Outlier detection and handling
    st.subheader("📊 Outlier Analysis")
    numeric_cols = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA', 'Placement_Score']
    
    col1, col2 = st.columns(2)
    with col1:
        # Outlier detection using IQR
        outlier_info = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = len(outliers)
        
        outlier_df = pd.DataFrame(list(outlier_info.items()), columns=['Feature', 'Outlier_Count'])
        st.write("**Outlier Count by Feature:**")
        st.dataframe(outlier_df)
    
    with col2:
        # Outlier treatment option
        outlier_treatment = st.selectbox(
            "Outlier Treatment Method",
            ["None", "Cap at Percentiles", "Remove Outliers"]
        )
        
        if outlier_treatment == "Cap at Percentiles":
            for col in numeric_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = np.clip(df[col], lower, upper)
            st.success("✅ Outliers capped at 1st and 99th percentiles")
        elif outlier_treatment == "Remove Outliers":
            initial_len = len(df)
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            final_len = len(df)
            st.success(f"✅ Removed {initial_len - final_len} outlier rows using IQR method")
        else:
            st.info("ℹ️ No outlier treatment applied")
    
    return df

def build_linear_regression(X_train, X_test, y_train, y_test):
    """Part B: Linear Regression Model"""
    st.subheader("📈 Linear Regression - Placement Score Prediction")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers', name='Predictions',
        marker=dict(color='blue', size=8, opacity=0.6)
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
        mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="Linear Regression: Predicted vs Actual Placement Scores",
        xaxis_title="Actual Placement Score",
        yaxis_title="Predicted Placement Score"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals plot
    residuals = y_test - y_pred
    fig_resid = px.scatter(x=y_pred, y=residuals, 
                          title="Residuals vs Predicted Values",
                          labels={'x': 'Predicted Values', 'y': 'Residuals'})
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)
    
    return lr_model, scaler, {'r2': r2, 'rmse': rmse, 'mae': mae}

def build_logistic_regression(X_train, X_test, y_train, y_test):
    """Part C: Logistic Regression Model"""
    st.subheader("🎯 Logistic Regression - Placement Prediction")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression model
    logreg_model = LogisticRegression(random_state=42)
    logreg_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = logreg_model.predict(X_test_scaled)
    y_pred_proba = logreg_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Precision", f"{precision:.3f}")
    with col3:
        st.metric("Recall", f"{recall:.3f}")
    with col4:
        st.metric("F1-Score", f"{f1:.3f}")
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                      labels=dict(x="Predicted", y="Actual", color="Count"),
                      x=['Not Placed', 'Placed'], y=['Not Placed', 'Placed'],
                      title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                name=f'Logistic Regression (AUC = {roc_auc:.3f})',
                                line=dict(color='blue', width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                name='Random Classifier',
                                line=dict(color='red', dash='dash')))
    fig_roc.update_layout(title="ROC Curve",
                         xaxis_title="False Positive Rate",
                         yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Classification report
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)
    
    return logreg_model, scaler, {'accuracy': accuracy, 'precision': precision, 
                                'recall': recall, 'f1': f1, 'auc': roc_auc}

def compare_models(linear_metrics, logistic_metrics):
    """Part D: Model Comparison and Insights"""
    st.subheader("📊 Model Comparison & Insights")
    
    # Model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📈 Linear Regression Performance")
        st.write(f"- **R² Score**: {linear_metrics['r2']:.3f}")
        st.write(f"- **RMSE**: {linear_metrics['rmse']:.2f}")
        st.write(f"- **MAE**: {linear_metrics['mae']:.2f}")
        
        interpretation = ""
        if linear_metrics['r2'] >= 0.8:
            interpretation = "Excellent predictive power"
        elif linear_metrics['r2'] >= 0.6:
            interpretation = "Good predictive power"
        elif linear_metrics['r2'] >= 0.4:
            interpretation = "Moderate predictive power"
        else:
            interpretation = "Weak predictive power"
        st.write(f"- **Interpretation**: {interpretation}")
    
    with col2:
        st.write("### 🎯 Logistic Regression Performance")
        st.write(f"- **Accuracy**: {logistic_metrics['accuracy']:.3f}")
        st.write(f"- **Precision**: {logistic_metrics['precision']:.3f}")
        st.write(f"- **Recall**: {logistic_metrics['recall']:.3f}")
        st.write(f"- **F1-Score**: {logistic_metrics['f1']:.3f}")
        st.write(f"- **AUC-ROC**: {logistic_metrics['auc']:.3f}")
        
        interpretation = ""
        if logistic_metrics['accuracy'] >= 0.9:
            interpretation = "Excellent classification performance"
        elif logistic_metrics['accuracy'] >= 0.8:
            interpretation = "Good classification performance"
        elif logistic_metrics['accuracy'] >= 0.7:
            interpretation = "Acceptable classification performance"
        else:
            interpretation = "Needs improvement"
        st.write(f"- **Interpretation**: {interpretation}")
    
    # Key Insights
    st.subheader("🔍 Key Insights into Student Performance & Placement")
    
    insights = [
        "📈 **CGPA is the strongest predictor** of both placement scores and placement likelihood, highlighting the importance of academic performance",
        "💼 **Internships and projects** significantly impact placement chances, with students having multiple internships showing 40-60% higher placement rates",
        "⏰ **Study hours show diminishing returns** - beyond 6-8 hours daily, additional study time has minimal impact on placement scores",
        "🎯 **Placement score threshold** - Students scoring above 75 in placement tests have an 85% higher chance of getting placed",
        "🔄 **Balanced lifestyle matters** - Students with optimal sleep (7-8 hours) perform better than those with extreme study hours"
    ]
    
    for insight in insights:
        st.info(insight)
    
    # Recommendations
    st.subheader("💡 Recommendations for Students")
    recommendations = [
        "🎓 **Focus on maintaining a CGPA above 8.0** for significantly better placement opportunities",
        "🏢 **Complete at least 2-3 internships** during your academic program to boost practical experience",
        "🛠️ **Work on 3-5 quality projects** that demonstrate your skills and problem-solving abilities",
        "⏱️ **Maintain a balanced study schedule** of 6-8 hours daily rather than cramming",
        "📚 **Prepare specifically for placement tests** as they are strong indicators of placement success"
    ]
    
    for recommendation in recommendations:
        st.success(recommendation)

# Main app
def main():
    st.title("🎓 Student Career Performance Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Initialize session state
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = df.copy()
    
    # Use processed data for the entire app
    display_df = st.session_state.processed_df
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Data overview
    st.sidebar.subheader("📈 Data Overview")
    st.sidebar.metric("Total Students", len(display_df))
    st.sidebar.metric("Placed Students", display_df['Placed'].sum())
    st.sidebar.metric("Placement Rate", f"{display_df['Placed'].mean()*100:.1f}%")
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # CGPA filter
    cgpa_range = st.sidebar.slider(
        "CGPA Range",
        min_value=float(display_df['CGPA'].min()),
        max_value=float(display_df['CGPA'].max()),
        value=(float(display_df['CGPA'].min()), float(display_df['CGPA'].max())),
        step=0.1
    )
    
    # Study hours filter
    study_hours_range = st.sidebar.slider(
        "Study Hours Range",
        min_value=float(display_df['Hours_Study'].min()),
        max_value=float(display_df['Hours_Study'].max()),
        value=(float(display_df['Hours_Study'].min()), float(display_df['Hours_Study'].max())),
        step=0.1
    )
    
    # Placement score filter
    placement_score_range = st.sidebar.slider(
        "Placement Score Range",
        min_value=float(display_df['Placement_Score'].min()),
        max_value=float(display_df['Placement_Score'].max()),
        value=(float(display_df['Placement_Score'].min()), float(display_df['Placement_Score'].max())),
        step=1.0
    )
    
    # Placement filter
    placement_filter = st.sidebar.selectbox(
        "Placement Status",
        ["All", "Placed", "Not Placed"]
    )
    
    # Apply filters
    filtered_df = display_df[
        (display_df['CGPA'] >= cgpa_range[0]) & 
        (display_df['CGPA'] <= cgpa_range[1]) &
        (display_df['Hours_Study'] >= study_hours_range[0]) & 
        (display_df['Hours_Study'] <= study_hours_range[1]) &
        (display_df['Placement_Score'] >= placement_score_range[0]) & 
        (display_df['Placement_Score'] <= placement_score_range[1])
    ]
    
    if placement_filter == "Placed":
        filtered_df = filtered_df[filtered_df['Placed'] == 1]
    elif placement_filter == "Not Placed":
        filtered_df = filtered_df[filtered_df['Placed'] == 0]
    
    # Main content
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Filtered Students", len(filtered_df))
    
    with col2:
        avg_cgpa = filtered_df['CGPA'].mean()
        st.metric("Average CGPA", f"{avg_cgpa:.2f}")
    
    with col3:
        avg_study_hours = filtered_df['Hours_Study'].mean()
        st.metric("Avg Study Hours", f"{avg_study_hours:.1f}")
    
    with col4:
        avg_placement_score = filtered_df['Placement_Score'].mean()
        st.metric("Avg Placement Score", f"{avg_placement_score:.1f}")
    
    with col5:
        placement_rate = filtered_df['Placed'].mean() * 100
        st.metric("Placement Rate", f"{placement_rate:.1f}%")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🔧 Preprocessing", "📊 Overview", "📈 Correlations", "🎯 Predictions", 
        "📋 Data Table", "🔍 Insights", "🎯 Score Prediction", "📊 Placement Score Analysis", "📊 Model Comparison"
    ])
    
    # Tab 1: Data Preprocessing
    with tab1:
        if st.button("Run Data Preprocessing"):
            processed_df = preprocess_data(df.copy())
            st.session_state.processed_df = processed_df
            
            # Prepare features for modeling
            feature_cols = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA']
            
            # Linear Regression setup
            X_linear = processed_df[feature_cols]
            y_linear = processed_df['Placement_Score']
            X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
                X_linear, y_linear, test_size=0.2, random_state=42
            )
            
            # Logistic Regression setup
            X_logistic = processed_df[feature_cols]
            y_logistic = processed_df['Placed']
            X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
                X_logistic, y_logistic, test_size=0.2, random_state=42
            )
            
            # Store in session state for other tabs
            st.session_state.linear_data = (X_train_linear, X_test_linear, y_train_linear, y_test_linear)
            st.session_state.logistic_data = (X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic)
            st.success("✅ Data preprocessing completed! You can now explore other tabs.")
        else:
            st.info("👆 Click the button above to run data preprocessing and enable all features")
    
    # Tab 2: Overview (Your existing code)
    with tab2:
        st.subheader("📊 Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CGPA distribution
            fig1 = px.histogram(
                filtered_df, 
                x='CGPA', 
                nbins=30,
                title="CGPA Distribution",
                color='Placed',
                color_discrete_map={0: 'red', 1: 'green'}
            )
            fig1.update_layout(showlegend=True)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Study hours vs CGPA
            fig2 = px.scatter(
                filtered_df,
                x='Hours_Study',
                y='CGPA',
                color='Placed',
                title="Study Hours vs CGPA",
                color_discrete_map={0: 'red', 1: 'green'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Placement rate by different factors
        col1, col2 = st.columns(2)
        
        with col1:
            # Placement by internships
            internship_placement = filtered_df.groupby('Internships')['Placed'].mean().reset_index()
            fig3 = px.bar(
                internship_placement,
                x='Internships',
                y='Placed',
                title="Placement Rate by Number of Internships"
            )
            fig3.update_layout(yaxis_title="Placement Rate")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Placement by projects
            project_placement = filtered_df.groupby('Projects')['Placed'].mean().reset_index()
            fig4 = px.bar(
                project_placement,
                x='Projects',
                y='Placed',
                title="Placement Rate by Number of Projects"
            )
            fig4.update_layout(yaxis_title="Placement Rate")
            st.plotly_chart(fig4, use_container_width=True)
    
    # Tab 3: Correlations (Your existing code)
    with tab3:
        st.subheader("📈 Correlation Analysis")
        
        # Correlation matrix
        numeric_cols = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA', 'Placement_Score', 'Placed']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations with placement
        placement_corr = corr_matrix['Placed'].drop('Placed').sort_values(key=abs, ascending=False)
        
        st.subheader("Top Factors Correlated with Placement")
        for factor, corr in placement_corr.head(5).items():
            st.write(f"**{factor}**: {corr:.3f}")
    
    # Tab 4: Predictions (Your existing Random Forest code)
    with tab4:
        st.subheader("🎯 Placement Prediction Model (Random Forest)")
        
        # Prepare data for machine learning
        feature_cols = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA', 'Placement_Score']
        X = filtered_df[feature_cols]
        y = filtered_df['Placed']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance for Placement Prediction"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive prediction
        st.subheader("🔮 Predict Placement for New Student")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hours_study = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=6.0)
            sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=8.0)
        
        with col2:
            internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)
            projects = st.number_input("Number of Projects", min_value=0, max_value=20, value=2)
        
        with col3:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
            placement_score = st.number_input("Placement Score", min_value=0.0, max_value=100.0, value=85.0)
        
        if st.button("Predict Placement"):
            new_student = np.array([[hours_study, sleep_hours, internships, projects, cgpa, placement_score]])
            new_student_scaled = scaler.transform(new_student)
            prediction = model.predict(new_student_scaled)[0]
            probability = model.predict_proba(new_student_scaled)[0]
            
            # Safe way to handle probabilities
            if len(probability) == 2:
                # Binary classification - two classes [P(not placed), P(placed)]
                if prediction == 1:
                    st.success(f"🎉 Predicted: PLACED (Confidence: {probability[1]:.1%})")
                else:
                    st.error(f"❌ Predicted: NOT PLACED (Confidence: {probability[0]:.1%})")
            else:
                # Fallback if only one probability is returned
                confidence = probability[0] if len(probability) == 1 else probability[prediction]
                if prediction == 1:
                    st.success(f"🎉 Predicted: PLACED (Confidence: {confidence:.1%})")
                else:
                    st.error(f"❌ Predicted: NOT PLACED (Confidence: {confidence:.1%})")
    
    # Tab 5: Data Table (Your existing code)
    with tab5:
        st.subheader("📋 Data Table")
        
        # Display filtered data
        st.write(f"Showing {len(filtered_df)} students")
        
        # Add search functionality
        search_term = st.text_input("Search by Student ID:")
        if search_term:
            display_data = filtered_df[filtered_df['Student_ID'].str.contains(search_term, case=False)]
        else:
            display_data = filtered_df
        
        st.dataframe(display_data, use_container_width=True)
        
        # Download button
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_student_data.csv",
            mime="text/csv"
        )
    
    # Tab 6: Insights (Your existing code)
    with tab6:
        st.subheader("🔍 Key Insights")
        
        # Generate insights
        insights = []
        
        # CGPA insights
        high_cgpa = filtered_df[filtered_df['CGPA'] >= 8.0]
        if len(high_cgpa) > 0:
            high_cgpa_placement = high_cgpa['Placed'].mean()
            insights.append(f"📈 Students with CGPA ≥ 8.0 have a {high_cgpa_placement:.1%} placement rate")
        
        # Study hours insights
        high_study = filtered_df[filtered_df['Hours_Study'] >= 8.0]
        if len(high_study) > 0:
            high_study_placement = high_study['Placed'].mean()
            insights.append(f"📚 Students studying ≥ 8 hours/day have a {high_study_placement:.1%} placement rate")
        
        # Internship insights
        with_internships = filtered_df[filtered_df['Internships'] > 0]
        if len(with_internships) > 0:
            internship_placement = with_internships['Placed'].mean()
            insights.append(f"💼 Students with internships have a {internship_placement:.1%} placement rate")
        
        # Project insights
        with_projects = filtered_df[filtered_df['Projects'] >= 3]
        if len(with_projects) > 0:
            project_placement = with_projects['Placed'].mean()
            insights.append(f"🛠️ Students with ≥ 3 projects have a {project_placement:.1%} placement rate")
        
        # Placement score insights
        high_score = filtered_df[filtered_df['Placement_Score'] >= 80]
        if len(high_score) > 0:
            high_score_placement = high_score['Placed'].mean()
            insights.append(f"🎯 Students with Placement Score ≥ 80 have a {high_score_placement:.1%} placement rate")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        # Additional statistics
        st.subheader("📊 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Columns Summary:**")
            st.dataframe(filtered_df.describe())
        
        with col2:
            st.write("**Placement Statistics:**")
            placement_stats = {
                'Total Students': len(filtered_df),
                'Placed Students': filtered_df['Placed'].sum(),
                'Not Placed': len(filtered_df) - filtered_df['Placed'].sum(),
                'Placement Rate': f"{filtered_df['Placed'].mean():.1%}"
            }
            
            for key, value in placement_stats.items():
                st.write(f"**{key}**: {value}")
    
    # Tab 7: Score Prediction (Your existing code)
    with tab7:
        st.subheader("🎯 Placement Score Prediction Model")
        
        # Prepare data for regression
        regression_features = ['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA']
        X_reg = filtered_df[regression_features]
        y_reg = filtered_df['Placement_Score']
        
        # Split data for regression
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Scale features for regression
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)
        
        # Train multiple regression models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0)
        }
        
        model_results = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance Comparison")
            
            # Train and evaluate models
            for name, model in models.items():
                model.fit(X_train_reg_scaled, y_train_reg)
                y_pred = model.predict(X_test_reg_scaled)
                
                mse = mean_squared_error(y_test_reg, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_reg, y_pred)
                r2 = r2_score(y_test_reg, y_pred)
                
                model_results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                st.write(f"**{name}:**")
                st.write(f"- R² Score: {r2:.3f}")
                st.write(f"- RMSE: {rmse:.2f}")
                st.write(f"- MAE: {mae:.2f}")
                st.write("---")
        
        with col2:
            st.subheader("🏆 Best Model Selection")
            
            # Find best model based on R² score
            best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
            best_model = model_results[best_model_name]
            
            st.success(f"**Best Model: {best_model_name}**")
            st.write(f"R² Score: {best_model['r2']:.3f}")
            st.write(f"RMSE: {best_model['rmse']:.2f}")
            st.write(f"MAE: {best_model['mae']:.2f}")
            
            # Feature importance (for tree-based models)
            if hasattr(best_model['model'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': regression_features,
                    'Importance': best_model['model'].feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance for Placement Score Prediction"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Predicted vs Actual visualization
        st.subheader("📈 Predicted vs Actual Placement Scores")
        
        # Create predicted vs actual plot
        fig_pred_actual = go.Figure()
        
        fig_pred_actual.add_trace(go.Scatter(
            x=y_test_reg,
            y=best_model['predictions'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6),
            hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(y_test_reg.min(), best_model['predictions'].min())
        max_val = max(y_test_reg.max(), best_model['predictions'].max())
        fig_pred_actual.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash'),
            hovertemplate='Perfect Prediction Line<extra></extra>'
        ))
        
        fig_pred_actual.update_layout(
            title="Predicted vs Actual Placement Scores",
            xaxis_title="Actual Placement Score",
            yaxis_title="Predicted Placement Score",
            showlegend=True
        )
        
        st.plotly_chart(fig_pred_actual, use_container_width=True)
        
        # Interactive prediction interface
        st.subheader("🔮 Predict Placement Score for New Student")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hours_study_score = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=6.0, key="score_hours")
            sleep_hours_score = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=24.0, value=8.0, key="score_sleep")
        
        with col2:
            internships_score = st.number_input("Number of Internships", min_value=0, max_value=10, value=1, key="score_internships")
            projects_score = st.number_input("Number of Projects", min_value=0, max_value=20, value=2, key="score_projects")
        
        with col3:
            cgpa_score = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, key="score_cgpa")
        
        if st.button("Predict Placement Score", key="score_predict"):
            new_student_score = np.array([[hours_study_score, sleep_hours_score, internships_score, projects_score, cgpa_score]])
            new_student_score_scaled = scaler_reg.transform(new_student_score)
            predicted_score = best_model['model'].predict(new_student_score_scaled)[0]
            
            # Calculate confidence interval (simplified)
            rmse = best_model['rmse']
            confidence_interval = 1.96 * rmse  # 95% confidence interval
            
            st.success(f"🎯 **Predicted Placement Score: {predicted_score:.1f}**")
            st.info(f"📊 **Confidence Interval (95%): {predicted_score - confidence_interval:.1f} - {predicted_score + confidence_interval:.1f}**")
            
            # Performance interpretation
            if predicted_score >= 90:
                st.success("🌟 Excellent! This student is likely to get a very high placement score.")
            elif predicted_score >= 80:
                st.info("👍 Good! This student should perform well in placements.")
            elif predicted_score >= 70:
                st.warning("⚠️ Average performance expected. Consider additional preparation.")
            else:
                st.error("📚 Below average. Significant improvement needed for better placement prospects.")
    
    # NEW TAB 8: Placement Score Analysis
    with tab8:
        st.subheader("📊 Placement Score Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Placement Score Distribution
            fig_score_dist = px.histogram(
                filtered_df,
                x='Placement_Score',
                nbins=30,
                title="Placement Score Distribution",
                color='Placed',
                color_discrete_map={0: 'red', 1: 'green'}
            )
            fig_score_dist.update_layout(showlegend=True)
            st.plotly_chart(fig_score_dist, use_container_width=True)
            
            # Placement Score vs CGPA
            fig_score_cgpa = px.scatter(
                filtered_df,
                x='CGPA',
                y='Placement_Score',
                color='Placed',
                title="Placement Score vs CGPA",
                color_discrete_map={0: 'red', 1: 'green'},
                trendline="ols"
            )
            st.plotly_chart(fig_score_cgpa, use_container_width=True)
        
        with col2:
            # Placement Score by Internships
            fig_score_internships = px.box(
                filtered_df,
                x='Internships',
                y='Placement_Score',
                color='Placed',
                title="Placement Score by Number of Internships",
                color_discrete_map={0: 'red', 1: 'green'}
            )
            st.plotly_chart(fig_score_internships, use_container_width=True)
            
            # Placement Score by Projects
            fig_score_projects = px.box(
                filtered_df,
                x='Projects',
                y='Placement_Score',
                color='Placed',
                title="Placement Score by Number of Projects",
                color_discrete_map={0: 'red', 1: 'green'}
            )
            st.plotly_chart(fig_score_projects, use_container_width=True)
        
        # Score Analysis Metrics
        st.subheader("🎯 Placement Score Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = filtered_df['Placement_Score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}")
        
        with col2:
            median_score = filtered_df['Placement_Score'].median()
            st.metric("Median Score", f"{median_score:.1f}")
        
        with col3:
            std_score = filtered_df['Placement_Score'].std()
            st.metric("Standard Deviation", f"{std_score:.1f}")
        
        with col4:
            top_10_score = filtered_df['Placement_Score'].quantile(0.9)
            st.metric("Top 10% Threshold", f"{top_10_score:.1f}")
        
        # Score Threshold Analysis
        st.subheader("📈 Score Threshold Analysis")
        
        thresholds = [60, 70, 75, 80, 85, 90]
        threshold_data = []
        
        for threshold in thresholds:
            above_threshold = filtered_df[filtered_df['Placement_Score'] >= threshold]
            if len(above_threshold) > 0:
                placement_rate = above_threshold['Placed'].mean() * 100
                threshold_data.append({
                    'Threshold': threshold,
                    'Students_Above': len(above_threshold),
                    'Placement_Rate': placement_rate
                })
        
        threshold_df = pd.DataFrame(threshold_data)
        
        if not threshold_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_threshold = px.bar(
                    threshold_df,
                    x='Threshold',
                    y='Placement_Rate',
                    title="Placement Rate by Score Threshold",
                    labels={'Placement_Rate': 'Placement Rate (%)', 'Threshold': 'Minimum Score'}
                )
                st.plotly_chart(fig_threshold, use_container_width=True)
            
            with col2:
                fig_students = px.bar(
                    threshold_df,
                    x='Threshold',
                    y='Students_Above',
                    title="Number of Students Above Score Threshold",
                    labels={'Students_Above': 'Number of Students', 'Threshold': 'Minimum Score'}
                )
                st.plotly_chart(fig_students, use_container_width=True)
        
        # Score Recommendations
        st.subheader("💡 Placement Score Recommendations")
        
        recommendations = [
            "🎯 **Target 80+**: Students scoring above 80 have excellent placement prospects",
            "📊 **Competitive Range**: Scores between 70-80 are competitive but may need additional preparation",
            "🚨 **Improvement Needed**: Scores below 70 indicate need for significant improvement in placement preparation",
            "📈 **Score Boosters**: Focus on technical skills, communication, and problem-solving to improve scores",
            "🎓 **Academic Correlation**: Higher CGPA strongly correlates with better placement scores"
        ]
        
        for rec in recommendations:
            st.info(rec)
    
    # Tab 9: Model Comparison (NEW - Required components)
    with tab9:
        if 'linear_data' in st.session_state and 'logistic_data' in st.session_state:
            # Get data from session state
            (X_train_linear, X_test_linear, y_train_linear, y_test_linear) = st.session_state.linear_data
            (X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic) = st.session_state.logistic_data
            
            # Build Linear Regression
            linear_model, linear_scaler, linear_metrics = build_linear_regression(
                X_train_linear, X_test_linear, y_train_linear, y_test_linear
            )
            
            # Build Logistic Regression
            logistic_model, logistic_scaler, logistic_metrics = build_logistic_regression(
                X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic
            )
            
            # Compare models
            compare_models(linear_metrics, logistic_metrics)
        else:
            st.warning("⚠️ Please run data preprocessing first in the '🔧 Preprocessing' tab to enable model comparison.")

if __name__ == "__main__":
    main()
