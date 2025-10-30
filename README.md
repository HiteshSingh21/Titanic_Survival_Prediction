
## 🚢 Titanic Survival Prediction

This project analyzes the classic Titanic dataset (`Titanic-Dataset.csv`) to predict passenger survival. The entire workflow, from data cleaning and feature engineering to model training and evaluation, is contained in the `Analysis.ipynb` Jupyter Notebook.

### Notebook Workflow

The notebook is structured to cover the following key steps:

1.  **Data Loading:** The `Titanic-Dataset.csv` file is loaded into a pandas DataFrame.
2.  **Data Cleaning & Feature Engineering:**
      * Handles missing data by imputing the mean for `Age` and dropping rows with missing `Embarked` data.
      * Engineers new features to improve model performance:
          * `Has_Cabin`: A binary feature indicating if a passenger had a cabin number listed.
          * `Deck`: Extracted from the `Cabin` feature (e.g., 'C', 'U' for Unknown).
          * `FamilySize`: A combination of `SibSp` (siblings/spouses) and `Parch` (parents/children).
          * `Title`: Extracted from the passenger's `Name` (e.g., Mr., Mrs., Miss., Master).
      * Drops irrelevant columns like `Ticket` and the original `Cabin`.
3.  **Exploratory Data Analysis (EDA):**
      * Uses `seaborn` and `matplotlib` to create visualizations (count plots, histograms) to understand the relationship between survival and features like `Pclass`, `Sex`, `Age`, `Deck`, and `Title`.
4.  **Pre-processing for Modeling:**
      * Performs one-hot encoding on all categorical features (`Sex`, `Embarked`, `Title`, `Deck`).
      * Splits the data into training and test sets.
      * Scales all features using `StandardScaler` to prepare them for the models.
5.  **Modeling & Evaluation:**
      * Trains and evaluates the accuracy of several different classification models:
          * Logistic Regression
          * Random Forest Classifier
          * Gradient Boosting Classifier
      * Builds an ensemble `VotingClassifier` (combining Logistic Regression, XGBClassifier, and Random Forest) to leverage the strengths of multiple models.
      * Uses `GridSearchCV` to find the best hyperparameters for the ensemble model.
      * The final model accuracy is printed after training and prediction.

### 🛠️ How to Run

To run this notebook and reproduce the analysis:

1.  Ensure you have Python 3.x installed.

2.  Install the core dependencies. A `requirements.txt` was not provided, but you can install the main libraries using pip:

    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn xgboost
    ```

3.  Once the dependencies are installed, you can launch the notebook:

    ```bash
    jupyter notebook Analysis.ipynb
    ```
