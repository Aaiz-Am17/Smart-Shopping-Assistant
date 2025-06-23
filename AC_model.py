# ac_model.py
import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

class ACModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.load_message = ""
        self.column_transformer = None
        self.ensemble_model = None
        self.rf_best_params = None
        self.rf_mean_r2 = None
        self.xgb_best_params = None
        self.xgb_mean_r2 = None
        self.ensemble_r2 = None

        self._load_and_train_model()

    def _load_dataset(self):
        try:
            self.df = pd.read_csv(self.file_path)
            self.load_message = "AC Dataset loaded successfully!"
            return True
        except Exception as e:
            self.load_message = f"Error loading AC dataset: {e}"
            return False

    def _preprocess_data(self, df_to_preprocess, for_prediction=False):
        # Ensure required columns exist
        required_columns = ['Power_Consumption', 'Noise_level', 'Refrigerant', 'Condenser_Coil']
        if not for_prediction:
            required_columns.append('Price')
        for col in required_columns:
            if col not in df_to_preprocess.columns:
                raise ValueError(f"Missing required column: {col}")

        # Clean 'Power_Consumption'
        df_to_preprocess['Power_Consumption'] = df_to_preprocess['Power_Consumption'].astype(str).str.replace(r'[^\d.]+', '', regex=True)
        df_to_preprocess['Power_Consumption'] = pd.to_numeric(df_to_preprocess['Power_Consumption'], errors='coerce')
        df_to_preprocess['Power_Consumption'].fillna(df_to_preprocess['Power_Consumption'].median(), inplace=True)

        # Clean 'Noise_level'
        df_to_preprocess['Noise_level'] = df_to_to_preprocess['Noise_level'].astype(str).str.extract(r'(\d+)')[0].astype(float)
        df_to_preprocess['Noise_level'].fillna(df_to_to_preprocess['Noise_level'].median(), inplace=True)

        # Handle 'Refrigerant' column
        df_to_preprocess['Refrigerant'] = df_to_to_preprocess['Refrigerant'].apply(lambda x: 'R-32' if 'R-32' in str(x) else
                                                          'R410a' if 'R410a' in str(x) else 'Other')
        df_to_preprocess['Refrigerant'].fillna(df_to_to_preprocess['Refrigerant'].mode()[0], inplace=True)

        # Handle 'Condenser_Coil' column
        df_to_preprocess['Condenser_Coil'].fillna(df_to_to_preprocess['Condenser_Coil'].mode()[0], inplace=True)

        return df_to_preprocess

    def _train_model(self):
        if self.df is None:
            return

        self.df = self._preprocess_data(self.df.copy()) # Using a copy to avoid modifying original df during preprocessing

        X = self.df[['Condenser_Coil', 'Refrigerant', 'Power_Consumption', 'Noise_level']]
        y = self.df['Price']

        categorical_features = ['Condenser_Coil', 'Refrigerant']
        numerical_features = ['Power_Consumption', 'Noise_level']

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('scaler', StandardScaler(), numerical_features)
            ],
            remainder='passthrough'
        )

        X_encoded = self.column_transformer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # RandomForest
        rf_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, 50],
            'min_samples_split': [2, 5, 10]
        }
        rf_random_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42), rf_param_grid,
            n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42
        )
        rf_random_search.fit(X_train, y_train)
        self.rf_best_params = rf_random_search.best_params_
        self.rf_mean_r2 = np.mean(cross_val_score(rf_random_search.best_estimator_, X_train, y_train, cv=5, scoring='r2'))

        # XGBoost
        xgb_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        xgb_random_search = RandomizedSearchCV(
            XGBRegressor(random_state=42), xgb_param_grid,
            n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42
        )
        xgb_random_search.fit(X_train, y_train)
        self.xgb_best_params = xgb_random_search.best_params_
        self.xgb_mean_r2 = np.mean(cross_val_score(xgb_random_search.best_estimator_, X_train, y_train, cv=5, scoring='r2'))

        # Ensemble model
        self.ensemble_model = VotingRegressor([('RandomForest', rf_random_search.best_estimator_), ('XGBoost', xgb_random_search.best_estimator_)])
        self.ensemble_model.fit(X_train, y_train)

        # Evaluate on test set
        self.ensemble_r2 = r2_score(y_test, self.ensemble_model.predict(X_test))

    def _load_and_train_model(self):
        if self._load_dataset():
            self._train_model()

    def predict_price(self, user_choices):
        if self.ensemble_model is None or self.column_transformer is None:
            raise ValueError("AC model is not trained or loaded properly.")

        # Map 'Low', 'Medium', 'High' for Power_Consumption and Noise_level to numerical values
        # These mappings should ideally be based on the min/median/max of the training data.
        power_map = {'Low': self.df['Power_Consumption'].min(), 'Medium': self.df['Power_Consumption'].median(), 'High': self.df['Power_Consumption'].max()}
        noise_map = {'Low': self.df['Noise_level'].min(), 'Medium': self.df['Noise_level'].median(), 'High': self.df['Noise_level'].max()}

        input_data = user_choices.copy()
        input_data['Power_Consumption'] = power_map.get(input_data['Power_Consumption'], self.df['Power_Consumption'].median())
        input_data['Noise_level'] = noise_map.get(input_data['Noise_level'], self.df['Noise_level'].median())

        # Create a DataFrame with the same columns as X used for training
        X_cols = ['Condenser_Coil', 'Refrigerant', 'Power_Consumption', 'Noise_level']
        input_df = pd.DataFrame(columns=X_cols)
        input_df.loc[0] = [input_data.get(col, self.df[col].mode()[0] if col in self.df.columns else None) for col in X_cols]

        input_encoded = self.column_transformer.transform(input_df)
        predicted_price = self.ensemble_model.predict(input_encoded)[0]
        return predicted_price

# Example Usage (for testing the module independently)
if __name__ == "__main__":
    AC_FILE_PATH = "Dataset/Air_condition_dataset.csv"
    ac_predictor = ACModel(AC_FILE_PATH)
    print(f"AC Load Message: {ac_predictor.load_message}")
    if ac_predictor.ensemble_model:
        print(f"AC RandomForest Best Params: {ac_predictor.rf_best_params}")
        print(f"AC RandomForest Mean R2: {ac_predictor.rf_mean_r2:.3f}")
        print(f"AC XGBoost Best Params: {ac_predictor.xgb_best_params}")
        print(f"AC XGBoost Mean R2: {ac_predictor.xgb_mean_r2:.3f}")
        print(f"AC Ensemble R2: {ac_predictor.ensemble_r2:.3f}")

        # Example prediction
        sample_choices = {
            'Condenser_Coil': 'Copper',
            'Refrigerant': 'R-32',
            'Power_Consumption': 'Medium',
            'Noise_level': 'Low'
        }
        try:
            predicted_price = ac_predictor.predict_price(sample_choices)
            print(f"Predicted AC Price for sample choices: {predicted_price:.2f}")
        except ValueError as e:
            print(f"Prediction error: {e}")
