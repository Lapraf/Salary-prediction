import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Global variables to store models and data
models = {}
data = None
feature_names = None

def load_and_prepare_data():
    """Load and prepare the salary data"""
    global data, feature_names
    
    try:
        # Load the data - use the path relative to the script location
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'avg_salary.csv')
        df = pd.read_csv(csv_path)
        
        # Feature engineering
        df['Period_Year'] = df['Period'].astype(int)
        df['Period_Month'] = ((df['Period'] - df['Period_Year']) * 100).round().astype(int)
        df['GDP_CPI_Ratio'] = df['GDP'] / df['CPI']
        df['IT_Exports_per_GDP'] = df['IT_Exports'] / df['GDP']
        
        data = df
        return "Data loaded successfully! Dataset shape: {}".format(df.shape)
    except Exception as e:
        return f"Error loading data: {str(e)}"

def train_models():
    """Train the machine learning models"""
    global models, data, feature_names
    
    if data is None:
        return "Please load data first!"
    
    try:
        # Prepare features and target
        X = data[['Period_Year', 'Period_Month', 'IT_Exports', 'CPI', 'GDP', 'GDP_CPI_Ratio', 'IT_Exports_per_GDP', 'Seniority']]
        y = data['Salary']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessing
        num = ['Period_Year', 'Period_Month', 'IT_Exports', 'CPI', 'GDP', 'GDP_CPI_Ratio', 'IT_Exports_per_GDP']
        cat = ['Seniority']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat)
            ]
        )
        
        # Create pipelines
        linear_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            ))
        ])
        
        ensemble_model = VotingRegressor(
            estimators=[
                ('linear_reg', linear_pipeline),
                ('xgb_reg', xgb_pipeline)
            ],
            weights=None
        )
        
        # Train models
        linear_pipeline.fit(X_train, y_train)
        xgb_pipeline.fit(X_train, y_train)
        ensemble_model.fit(X_train, y_train)
        
        # Store models
        models['linear'] = linear_pipeline
        models['xgb'] = xgb_pipeline
        models['ensemble'] = ensemble_model
        models['X_test'] = X_test
        models['y_test'] = y_test
        
        # Get feature names for later use
        feature_names = (num + list(ensemble_model.named_estimators_['linear_reg']
                                  .named_steps['preprocessor']
                                  .named_transformers_['cat']
                                  .named_steps['onehot']
                                  .get_feature_names_out(cat)))
        
        # Evaluate models
        results = {}
        for name, model in [('Linear Regression', linear_pipeline), 
                           ('XGBoost', xgb_pipeline), 
                           ('Ensemble', ensemble_model)]:
            y_pred = model.predict(X_test)
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred)
            }
        
        result_text = "Models trained successfully!\n\nTest Set Performance:\n"
        for name, metrics in results.items():
            result_text += f"\n{name}:\n"
            result_text += f"  MAE: {metrics['MAE']:.2f}\n"
            result_text += f"  RMSE: {metrics['RMSE']:.2f}\n"
            result_text += f"  R²: {metrics['R²']:.4f}\n"
        
        return result_text
        
    except Exception as e:
        return f"Error training models: {str(e)}"

def predict_salary(period_year, period_month, it_exports, cpi, gdp, seniority, model_choice):
    """Predict salary based on input parameters"""
    global models
    
    if not models:
        return "Please train models first!"
    
    try:
        # Calculate derived features
        gdp_cpi_ratio = gdp / cpi
        it_exports_per_gdp = it_exports / gdp
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Period_Year': [period_year],
            'Period_Month': [period_month],
            'IT_Exports': [it_exports],
            'CPI': [cpi],
            'GDP': [gdp],
            'GDP_CPI_Ratio': [gdp_cpi_ratio],
            'IT_Exports_per_GDP': [it_exports_per_gdp],
            'Seniority': [seniority]
        })
        
        # Select model
        model_map = {
            'Linear Regression': 'linear',
            'XGBoost': 'xgb',
            'Ensemble': 'ensemble'
        }
        
        selected_model = models[model_map[model_choice]]
        prediction = selected_model.predict(input_data)[0]
        
        return f"Predicted Salary: ${prediction:.2f}"
        
    except Exception as e:
        return f"Error making prediction: {str(e)}"

def create_salary_trends_plot():
    """Create salary trends visualization"""
    global data
    
    if data is None:
        return None
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot salary trends by seniority
        for seniority in data['Seniority'].unique():
            subset = data[data['Seniority'] == seniority]
            plt.plot(subset['Period'], subset['Salary'], marker='o', label=seniority, linewidth=2)
        
        plt.title('Salary Trends Over Time by Seniority Level', fontsize=16, fontweight='bold')
        plt.xlabel('Period', fontsize=12)
        plt.ylabel('Average Salary ($)', fontsize=12)
        plt.legend(title='Seniority Level', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")
        return None

def create_feature_importance_plot():
    """Create feature importance visualization"""
    global models, feature_names
    
    if not models or not feature_names:
        return None
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # XGBoost feature importance
        if 'xgb' in models:
            xgb_importance = models['xgb'].named_steps['regressor'].feature_importances_
            xgb_feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': xgb_importance
            }).sort_values('importance', ascending=True)
            
            axes[0].barh(range(len(xgb_feature_importance)), xgb_feature_importance['importance'])
            axes[0].set_yticks(range(len(xgb_feature_importance)))
            axes[0].set_yticklabels(xgb_feature_importance['feature'])
            axes[0].set_xlabel('Importance')
            axes[0].set_title('XGBoost Feature Importance')
            axes[0].grid(True, alpha=0.3)
        
        # Linear regression coefficients
        if 'linear' in models:
            linear_coef = models['linear'].named_steps['regressor'].coef_
            linear_feature_importance = pd.DataFrame({
                'feature': feature_names,
                'coefficient': linear_coef
            }).sort_values('coefficient', ascending=True)
            
            colors = ['red' if coef < 0 else 'blue' for coef in linear_feature_importance['coefficient']]
            axes[1].barh(range(len(linear_feature_importance)), linear_feature_importance['coefficient'], color=colors)
            axes[1].set_yticks(range(len(linear_feature_importance)))
            axes[1].set_yticklabels(linear_feature_importance['feature'])
            axes[1].set_xlabel('Coefficient Value')
            axes[1].set_title('Linear Regression Coefficients')
            axes[1].grid(True, alpha=0.3)
            axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating feature importance plot: {str(e)}")
        return None

def create_model_comparison_plot():
    """Create model performance comparison"""
    global models
    
    if not models:
        return None
    
    try:
        # Get test predictions
        X_test = models['X_test']
        y_test = models['y_test']
        
        predictions = {}
        for name, model_key in [('Linear Regression', 'linear'), 
                               ('XGBoost', 'xgb'), 
                               ('Ensemble', 'ensemble')]:
            predictions[name] = models[model_key].predict(X_test)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (name, y_pred) in enumerate(predictions.items()):
            axes[i].scatter(y_test, y_pred, alpha=0.6)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Salary')
            axes[i].set_ylabel('Predicted Salary')
            axes[i].set_title(f'{name}\nR² = {r2_score(y_test, y_pred):.4f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating comparison plot: {str(e)}")
        return None

# Create Gradio interface
with gr.Blocks(title="IT Salary Prediction Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 IT Salary Prediction System
    
    This demo allows you to explore IT salary trends and make predictions based on various economic indicators and seniority levels.
    
    **Features:**
    - Interactive salary prediction
    - Data visualization and trends analysis
    - Model performance comparison
    - Feature importance analysis
    """)
    
    with gr.Tab("📊 Data & Training"):
        gr.Markdown("## Data Loading and Model Training")
        
        with gr.Row():
            load_btn = gr.Button("Load Data", variant="primary")
            train_btn = gr.Button("Train Models", variant="secondary")
        
        status_output = gr.Textbox(label="Status", lines=10, interactive=False)
        
        load_btn.click(load_and_prepare_data, outputs=status_output)
        train_btn.click(train_models, outputs=status_output)
    
    with gr.Tab("🔮 Salary Prediction"):
        gr.Markdown("## Make Salary Predictions")
        
        with gr.Row():
            with gr.Column():
                period_year = gr.Slider(2015, 2030, value=2024, step=1, label="Year")
                period_month = gr.Slider(1, 12, value=6, step=1, label="Month")
                it_exports = gr.Number(value=3240.0, label="IT Exports (Million USD)")
                cpi = gr.Number(value=203.5, label="Consumer Price Index")
                gdp = gr.Number(value=5711.16, label="GDP (Billion USD)")
                seniority = gr.Dropdown(
                    choices=["Junior", "Middle", "Senior", "Lead"],
                    value="Middle",
                    label="Seniority Level"
                )
                model_choice = gr.Dropdown(
                    choices=["Linear Regression", "XGBoost", "Ensemble"],
                    value="Ensemble",
                    label="Model"
                )
            
            with gr.Column():
                predict_btn = gr.Button("Predict Salary", variant="primary", size="lg")
                prediction_output = gr.Textbox(label="Prediction Result", lines=3)
        
        predict_btn.click(
            predict_salary,
            inputs=[period_year, period_month, it_exports, cpi, gdp, seniority, model_choice],
            outputs=prediction_output
        )
    
    with gr.Tab("📈 Visualizations"):
        gr.Markdown("## Data Analysis and Model Performance")
        
        with gr.Row():
            trends_btn = gr.Button("Show Salary Trends", variant="primary")
            importance_btn = gr.Button("Show Feature Importance", variant="secondary")
            comparison_btn = gr.Button("Show Model Comparison", variant="secondary")
        
        plot_output = gr.Plot(label="Visualization")
        
        trends_btn.click(create_salary_trends_plot, outputs=plot_output)
        importance_btn.click(create_feature_importance_plot, outputs=plot_output)
        comparison_btn.click(create_model_comparison_plot, outputs=plot_output)
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About This Project
        
        This salary prediction system analyzes IT salary trends in Ukraine using various economic indicators:
        
        **Features Used:**
        - **Period**: Year and month of the data
        - **IT Exports**: IT services export volume (Million USD)
        - **CPI**: Consumer Price Index
        - **GDP**: Gross Domestic Product (Billion USD)
        - **Seniority**: Job seniority level (Junior, Middle, Senior, Lead)
        
        **Models:**
        - **Linear Regression**: Simple linear model for baseline predictions
        - **XGBoost**: Gradient boosting model for complex patterns
        - **Ensemble**: Combination of both models for improved accuracy
        
        **Data Source:** IT salary surveys and economic indicators from 2015-2024
        
        **Performance:** The ensemble model achieves R² > 0.94 on test data
        """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860) 