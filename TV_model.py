# tv_model.py
import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

class TVModel:
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

        self._load_and_train_model()

    def _load_dataset(self):
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.file_path, encoding=encoding)
                self.load_message = f"TV Dataset successfully read with encoding: {encoding}"
                return True
            except UnicodeDecodeError:
                pass
        self.load_message = "Unable to decode TV file with the specified encodings."
        return False

    def _preprocess_data(self, df_to_preprocess):
        df_to_preprocess.fillna(0, inplace=True) # Filling missing values

        df_to_preprocess['TV_OS_Category'] = df_to_preprocess['Operating_system'].apply(
            lambda x: 'Android' if 'Android' in str(x) else
                      'Linux' if 'Linux' in str(x) else
                      'Google TV' if 'Google TV' in str(x) else 'Other'
        )
        df_to_preprocess['TV_Picture_Quality_Category'] = df_to_preprocess['Picture_quality'].apply(
            lambda x: '4K' if '4K' in str(x) else
                      'Full HD' if 'Full HD' in str(x) else
                      'HD Ready' if 'HD Ready' in str(x) else 'Other'
        )
        # Ensuring 'Speaker' column is treated as string before extracting digits
        df_to_preprocess['TV_Speaker_Output_Category'] = df_to_preprocess['Speaker'].astype(str).str.extract(r'(\d+)')[0].astype(float).apply(
            lambda x: '10-30W' if 10 <= x <= 30 else
                      '30-60W' if 30 < x <= 60 else
                      '60-90W' if 60 < x <= 90 else '90+W'
        )
        return df_to_preprocess

    def _train_model(self):
        if self.df is None:
            return

        self.df = self._preprocess_data(self.df.copy()) # Using a copy

        X = self.df[['TV_OS_Category', 'TV_Picture_Quality_Category', 'TV_Speaker_Output_Category', 'Frequency', 'channel']]
        y = self.df['current_price']

        categorical_features = ['TV_OS_Category', 'TV_Picture_Quality_Category', 'TV_Speaker_Output_Category', 'Frequency', 'channel']
        self.column_transformer = ColumnTransformer(
            transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
            remainder='passthrough'
        )

        X_encoded = self.column_transformer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # RandomizedSearchCV for RandomForest
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        rf_random_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42), rf_param_grid,
            n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42
        )
        rf_random_search.fit(X_train, y_train)
        self.rf_best_params = rf_random_search.best_params_
        self.rf_mean_r2 = np.mean(cross_val_score(rf_random_search.best_estimator_, X_train, y_train, cv=5, scoring='r2'))

        # RandomizedSearchCV for XGBoost
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_child_weight': [1, 3, 5]
        }
        xgb_random_search = RandomizedSearchCV(
            XGBRegressor(random_state=42), xgb_param_grid,
            n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42
        )
        xgb_random_search.fit(X_train, y_train)
        self.xgb_best_params = xgb_random_search.best_params_
        self.xgb_mean_r2 = np.mean(cross_val_score(xgb_random_search.best_estimator_, X_train, y_train, cv=5, scoring='r2'))

        # Train Ensemble Model
        self.ensemble_model = VotingRegressor([('RandomForest', rf_random_search.best_estimator_), ('XGBoost', xgb_random_search.best_estimator_)])
        self.ensemble_model.fit(X_train, y_train)

    def _load_and_train_model(self):
        if self._load_dataset():
            self._train_model()

    def predict_price(self, user_choices):
        if self.ensemble_model is None or self.column_transformer is None:
            raise ValueError("TV model is not trained or loaded properly.")

        # Create a DataFrame from user choices
        user_data = pd.DataFrame({k: [v] for k, v in user_choices.items()})

        # Transform user data using the trained transformer
        user_data_encoded = self.column_transformer.transform(user_data)

        # Predict price using the ensemble model
        price = self.ensemble_model.predict(user_data_encoded)[0]
        return price

# Example Usage (for testing the module independently)
if __name__ == "__main__":
    TV_FILE_PATH = "Dataset/TELEVISION.csv"
    tv_predictor = TVModel(TV_FILE_PATH)
    print(f"TV Load Message: {tv_predictor.load_message}")
    if tv_predictor.ensemble_model:
        print(f"TV RandomForest Best Params: {tv_predictor.rf_best_params}")
        print(f"TV RandomForest Mean R2: {tv_predictor.rf_mean_r2:.3f}")
        print(f"TV XGBoost Best Params: {tv_predictor.xgb_best_params}")
        print(f"TV XGBoost Mean R2: {tv_predictor.xgb_mean_r2:.3f}")

        # Example prediction
        sample_choices = {
            'TV_OS_Category': 'Android',
            'TV_Picture_Quality_Category': '4K',
            'TV_Speaker_Output_Category': '10-30W',
            'Frequency': '60Hz',
            'channel': 'Netflix'
        }
        try:
            predicted_price = tv_predictor.predict_price(sample_choices)
            print(f"Predicted TV Price for sample choices: {predicted_price:.2f}")
        except ValueError as e:
            print(f"Prediction error: {e}")
