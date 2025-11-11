Here's a short README for your project.

-----

# Telco Customer Churn Prediction

This project analyzes the "WA\_Fn-UseC\_-Telco-Customer-Churn.csv" dataset to build a model that predicts customer churn. It includes a Jupyter Notebook for analysis and a Streamlit app for real-time predictions.

## Files

  * **`Customer_Churn_Prediction.ipynb`**:

      * Loads and visualizes the dataset.
      * Preprocesses the data: drops `customerID`, converts `TotalCharges` to numeric and fills nulls, and applies `LabelEncoder` to all categorical columns.
      * Handles the imbalanced dataset using **SMOTE** (Synthetic Minority Over-sampling Technique).
      * Splits the resampled data into training and test sets.
      * Trains and compares three models: Decision Tree, Random Forest, and XGBoost.
      * Selects **Random Forest** as the best model (85.8% accuracy) and saves it as `customer_churn_model.pkl` using `pickle`.
      * Saves the list of feature names as `feature_names.pkl`.

  * **`app.py`**:

      * A **Streamlit** web application that loads the saved `customer_churn_model.pkl` and `feature_names.pkl`.
      * Provides a user interface with dropdowns and number inputs for all 19 features (e.g., `tenure`, `Contract`, `MonthlyCharges`).
      * Preprocesses the live user input to match the model's training format.
      * On clicking the "Predict Churn" button, it displays the prediction ("Likely to Churn" or "Not Likely to Churn") along with a confidence score.

-----

## How to Run

**1. Install Dependencies**
You need the following Python libraries.

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit
```

**2. Generate the Model**

1.  Place the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the same directory.
2.  Run the `Customer_Churn_Prediction.ipynb` notebook from top to bottom.
3.  This will create `customer_churn_model.pkl` and `feature_names.pkl` in your folder.

**3. Run the App**

1.  In your terminal, run the following command:
    ```bash
    streamlit run app.py
    ```
2.  Your web browser will automatically open with the interactive prediction app.
