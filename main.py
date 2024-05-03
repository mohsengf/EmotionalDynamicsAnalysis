import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator

class BaseExcelModel:
    def __init__(self, file_path, engine='openpyxl'):
        self.file_path = file_path
        self.engine = engine
        self.data = pd.read_excel(file_path, sheet_name=None, engine=engine)

    def process_sheet(self, df, model, sheet_name):
        # Flexible column detection based on sheet content
        feature_column = 'Time' if 'Time' in df.columns else 'Depth of Love'
        x = df[feature_column].values.reshape(-1, 1)
        df['Average'] = df[['Rep1', 'Rep2', 'Rep3']].mean(axis=1)
        y = df['Average'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Cross-validation
        cv_scores = cross_val_score(model, x_train, y_train, cv=4, scoring='neg_mean_absolute_error')
        cv_mae = -np.mean(cv_scores)  # Convert negative MAE to positive
        print(f"Cross-validated Mean Absolute Error for {sheet_name}: {cv_mae}")

        model.fit(x_train, y_train)  # Fit model with training data
        predictions = model.predict(x_test)  # Predict with testing data
        test_mae = mean_absolute_error(y_test, predictions)
        print(f"Test Mean Absolute Error for {sheet_name}: {test_mae}")

        self.plot_data_and_predictions(x_train, y_train, x_test, predictions, sheet_name)

    def plot_data_and_predictions(self, x_train, y_train, x_test, predictions, sheet_name):
        plt.figure()
        plt.scatter(x_train, y_train, color='blue', label='Training data')
        plt.scatter(x_test, predictions, color='red', label='Predictions')
        plt.title(f"{sheet_name} - Model Predictions")
        plt.legend()
        plt.show()

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class RandomForestExcelModel(BaseExcelModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = RandomForestRegressor()

class SVRExcelModel(BaseExcelModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf'))
        ])

class KNNExcelModel(BaseExcelModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ])

class PolynomialExcelModel(BaseExcelModel):
    def __init__(self, file_path, degree=2):
        super().__init__(file_path)
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])

class NeuralNetworkExcelModel(BaseExcelModel):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)


# Example usage for a Random Forest model:
random_forest_model = RandomForestExcelModel('Three_Models_Dataset.xlsx')
for sheet_name, df in random_forest_model.data.items():
    random_forest_model.process_sheet(df, random_forest_model.model, sheet_name)
