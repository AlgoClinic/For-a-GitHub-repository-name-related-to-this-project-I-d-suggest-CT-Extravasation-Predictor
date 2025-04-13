# For-a-GitHub-repository-name-related-to-this-project-I-d-suggest-CT-Extravasation-Predictor
# Predicting Contrast Media Extravasation during CT: Development and Validation of a Machine Learning Model with SHAP Explainability

## Overview

This repository contains the complete codebase for our machine learning approach to predict contrast media extravasation risk during computed tomography (CT) examinations. The model integrates clinical, procedural, and patient-specific factors to provide personalized risk assessment, which may help prevent extravasation events and their associated complications.

## Research Background

Contrast media extravasation is a common adverse event during CT examinations that involves the leakage of contrast material into surrounding tissues. This can lead to significant patient discomfort, tissue damage, and in severe cases, compartment syndrome requiring surgical intervention. Early identification of high-risk patients could allow for preventive measures and personalized injection protocols.

Our model was developed using data from 500 patients who underwent contrast-enhanced CT examinations, including 252 cases with extravasation and 248 controls without extravasation. Through a rigorous feature selection process and comprehensive model evaluation, we have developed a clinically applicable prediction tool with transparent decision-making logic.

## Code Structure

The codebase is organized into a single comprehensive Python script with multiple functional components:

1. **Data Preprocessing** (Part 1)
   - Feature encoding and normalization
   - Data splitting and stratification
   - Outlier detection and multicollinearity assessment

2. **Descriptive Statistics** (Part 2)
   - Generation of clinical Table 1
   - Statistical comparison between groups
   - Appropriate test selection based on data characteristics

3. **Feature Selection** (Part 3)
   - Three-stage approach: univariate analysis → logistic regression → LASSO
   - Feature importance quantification
   - Dimension reduction while preserving predictive power

4. **LASSO Visualization** (Part 4)
   - Coefficient path plots
   - Cross-validation error curves
   - Feature stability assessment

5. **Model Training and Evaluation** (Part 5)
   - Comparison of multiple algorithms
   - Hyperparameter optimization
   - Performance metrics and statistical comparison (DeLong test)

6. **Model Performance Validation** (Part 6)
   - ROC curve analysis
   - Calibration assessment
   - Decision curve analysis

7. **Model Explanation with SHAP** (Part 7)
   - Global feature importance
   - Feature impact distribution
   - Individual prediction explanations

8. **Main Workflow** (Part 8)
   - End-to-end pipeline execution
   - Command-line interface
   - Model persistence

## Requirements

The code requires the following Python packages:
- pandas>=1.3.0
- numpy>=1.20.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- statsmodels>=0.13.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- xgboost>=1.5.0
- shap>=0.40.0

You can install all requirements using:
```
pip install -r requirements.txt
```

## Usage

### Basic Execution

To run the complete workflow with default settings:
```
python extravasation_model.py
```

### Specifying Data File and Output Directory

To specify a custom data file and output directory:
```
python extravasation_model.py --data_file "/path/to/your/data.csv" --save_dir "/path/to/save/results"
```

### Data Format

The input data should be a CSV file with the following columns:
- `group`: Target variable (1 = extravasation, 0 = no extravasation)
- Binary features: chemotherapy_history, needle_type, needle_insertion_depth, other_puncture, gender, malignancy, diabetes, hypertension, needle_dwell_time, needle_prior_use, pre_procedure_pain, needle_placement, patient_origin
- Categorical features: vein_quality, needle_site, needle_gauge, active_ingredient
- Continuous features: temp_diff, age, bmi, contrast_dose, injection_rate

## Output Files

The script generates various output files:

### Data Processing
- `processed_train_data.csv`, `processed_test_data.csv`: Preprocessed datasets
- `boxplot_continuous_by_group.png`: Visualization of continuous variables by group

### Statistical Analysis
- `table1_descriptive_statistics_full.csv`: Table 1 for the full dataset

### Feature Selection
- `selected_features_lasso.csv`: LASSO-selected features and coefficients
- `lasso_coefficient_path.png`: LASSO coefficient paths
- `lasso_cv_error.png`: LASSO cross-validation error

### Model Evaluation
- `model_performance_comparison_full.csv`: Performance metrics for all models
- `delong_test_results.csv`: Statistical comparison between models
- `roc_curve_comparison_train.png`, `roc_curve_comparison_test.png`: ROC curves

### Model Validation
- `cv_performance.csv`, `test_performance.csv`: Detailed validation metrics
- `roc_curve_combined.png`: Combined ROC curves for training and test sets
- `calibration_curve_combined.png`: Calibration curves
- `decision_curve_combined.png`: Decision curve analysis

### Model Explanation
- `shap_values.pkl`: Saved SHAP values
- `shap_feature_importance.png`: Global feature importance
- `shap_values_distribution.png`: SHAP value distributions
- `shap_force_plot_high_risk.png`, `shap_force_plot_low_risk.png`: Individual explanations
- `shap_feature_contributions.csv`: Feature contribution table

### Final Model
- `final_model.pkl`: Serialized final model

## Citation

If you use this code in your research, please cite our paper:

```
Author, A., Author, B., Author, C. (2025). Predicting Contrast Media Extravasation during CT: 
Development and Validation of a Machine Learning Model with SHAP Explainability. Radiology, XX(X), XXX-XXX.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact [your contact information].
