# timeseries_price_optimizer.py
import pandas as pd
import numpy as np
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import SlidingWindowSplitter, ForecastingGridSearchCV
from sktime.forecasting.base import ForecastingHorizon
from typing import List, Dict, Any, Optional, Union
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error  # Already used in test_model
from typing import List, Dict, Any, Optional, Union

class TimeSeriesPriceOptimizer:

    TRAIN_SIZE = 400  # Number of rows used for training
    DEFAULT_WINDOW_LENGTH = 110 
    DEFAULT_STEP_LENGTH = 10  

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        exog_col: str,
        date_col: str,
        price_col: str,
        window_length: int = None,
        step_length: int = None,
        train_size: int = None
    ):
        """
        Initialize the time series price optimizer.

        Args:
            data (pd.DataFrame): DataFrame containing the time series data.
            target_col (str): Column name of the target variable (e.g., demand).
            exog_col (str): Column name of the exogenous variable (e.g., price delta).
            date_col (str): Column name of the date/time index.
            price_col (str): Column name of the price variable for optimization.
            window_length (int, optional): Length of the sliding window for cross-validation.
            step_length (int, optional): Step length for the sliding window.
            train_size (int, optional): Number of rows to use for training.
        """
        self.data = data.copy()
        self.target_col = target_col
        self.exog_col = exog_col
        self.date_col = date_col
        self.price_col = price_col
        self.window_length = window_length if window_length is not None else self.DEFAULT_WINDOW_LENGTH
        self.step_length = step_length if step_length is not None else self.DEFAULT_STEP_LENGTH
        self.train_size = train_size if train_size is not None else self.TRAIN_SIZE
        self.forecaster = None
        self.best_params = None
        self.target_series = None
        self.exog_series = None
        self.dates = None
        self.price_series = None
        self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> None:
        """Load and preprocess the time series dataset."""

        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])


        self.data = self.data.sort_values(self.date_col)


        self.target_series = pd.Series(
            self.data[self.target_col].iloc[:self.train_size],
            index=self.data[self.date_col].iloc[:self.train_size]
        )
        self.exog_series = pd.Series(
            self.data[self.exog_col].iloc[:self.train_size],
            index=self.data[self.date_col].iloc[:self.train_size]
        )
        self.price_series = pd.Series(
            self.data[self.price_col].iloc[:self.train_size],
            index=self.data[self.date_col].iloc[:self.train_size]
        )
        self.dates = self.data[self.date_col].iloc[:self.train_size]

    def train_model(self, max_p: int = 5, max_q: int = 5, max_d: int = 1) -> Dict[str, Any]:
        """
        Train the ARIMA model using grid search over specified parameter ranges.

        Args:
            max_p (int): Maximum AR order.
            max_q (int): Maximum MA order.
            max_d (int): Maximum differencing order.

        Returns:
            Dict containing the best parameters and score.
        """
        fh = ForecastingHorizon(1)
        cv = SlidingWindowSplitter(
            start_with_window=True,
            fh=fh,
            step_length=self.step_length,
            window_length=self.window_length
        )


        arima_orders = [
            (p, d, q) for p in range(1, max_p + 1)
            for q in range(1, max_q + 1)
            for d in range(max_d + 1)
        ]
        param_grid = {"order": arima_orders}


        forecaster = ARIMA(
            enforce_invertibility=False,
            enforce_stationarity=False,
            suppress_warnings=True
        )


        grid_searcher = ForecastingGridSearchCV(
            forecaster=forecaster,
            cv=cv,
            param_grid=param_grid,
            error_score='raise',
            backend="multiprocessing"
        )


        grid_searcher.fit(self.target_series, X=self.exog_series.fillna(0))
        self.forecaster = grid_searcher.best_forecaster_
        self.best_params = {
            'order': grid_searcher.best_forecaster_.order,
            'score': grid_searcher.best_score_,
            'params': grid_searcher.best_forecaster_.get_fitted_params()
        }

        return self.best_params

    def predict(self, horizon: int, exog: pd.Series = None) -> pd.Series:
        """
        Predict the target variable for the specified horizon.

        Args:
            horizon (int): Number of periods to forecast.
            exog (pd.Series, optional): Exogenous variables for forecasting.

        Returns:
            pd.Series with the predicted values.
        """
        if self.forecaster is None:
            raise ValueError("Model must be trained before making predictions. Call train_model() first.")

        fh = ForecastingHorizon(range(1, horizon + 1))
        predictions = self.forecaster.predict(fh=fh, X=exog)
        return predictions

    def get_optimal_price(self, n_periods: int, historical_prices: pd.Series) -> List[float]:
        """
        Calculate optimal prices for the specified number of periods.

        Args:
            n_periods (int): Number of periods to calculate optimal prices for.
            historical_prices (pd.Series): Historical prices for the target item.

        Returns:
            List of optimal prices.
        """
        if self.forecaster is None:
            raise ValueError("Model must be trained before calculating optimal prices. Call train_model() first.")

        p, d, q = self.forecaster.order
        best_prices = []

        for t in range(n_periods, 1, -1):

            ar_features = [self.target_series.diff()[-i - t] for i in range(1, p + 1)]
            ma_features = [
                self.forecaster._forecaster.fittedvalues().diff()[-i - t] -
                self.target_series.diff()[-i - t] for i in range(1, q + 1)
            ]
            x = ar_features + ma_features


            ar_coefs = [self.forecaster.get_fitted_params()[f"ar.L{i}"] for i in range(1, p + 1)]
            ma_coefs = [self.forecaster.get_fitted_params()[f"ma.L{i}"] for i in range(1, q + 1)]
            coefs = ar_coefs + ma_coefs


            pr = self._get_optimal_price_linear(
                price_lag_1=historical_prices.iloc[len(self.target_series) - t - 1],
                x=x,
                intercept=self.forecaster.get_fitted_params()['intercept'],
                alpha=self.forecaster.get_fitted_params()[self.exog_col],
                a=coefs
            )
            best_prices.append(pr)

        return best_prices

    def _get_optimal_price_linear(
        self,
        a: List[float],
        intercept: float,
        alpha: float,
        price_lag_1: float,
        x: List[float]
    ) -> float:
        """
        Calculate the optimal price using a linear demand model.

        Args:
            a (List[float]): Coefficients of the best forecaster (without exogenous variable).
            intercept (float): Intercept value.
            alpha (float): Coefficient associated with the exogenous variable.
            price_lag_1 (float): Price in the previous period.
            x (List[float]): Features in the same order as in a.

        Returns:
            Optimal price for the next period.
        """
        best_delta = (-intercept - np.dot(a, x) - alpha) / (2 * alpha)
        return price_lag_1 * (1 + best_delta)
    
    def plot_predictions(self, predictions: pd.Series, true_values: pd.Series, save_path: str = None) -> None:
        """plot predicted vs true values"""
        import matplotlib.pyplot as plt
        plt.plot(predictions, color='b', label='predicted')
        plt.plot(true_values, color='g', label='true')
        plt.xlabel('observation')
        plt.ylabel('target value')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_residuals(self, predictions: pd.Series, true_values: pd.Series, save_path: str = None) -> None:
        """plot residuals of predictions"""
        import matplotlib.pyplot as plt
        resid = predictions - true_values
        plt.plot(resid / true_values, color='b', label='residuals')
        plt.title('residuals')
        plt.ylabel('error %')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_residual_histogram(self, predictions: pd.Series, true_values: pd.Series, bins: int = 20, save_path: str = None) -> None:
        """plot histogram of residuals"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        resid = predictions - true_values
        sns.histplot(resid, bins=bins)
        plt.xticks(rotation=70)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_residual_autocorrelation(self, predictions: pd.Series, true_values: pd.Series, save_path: str = None) -> None:
        """plot autocorrelation of residuals"""
        from statsmodels.graphics.tsaplots import plot_acf
        import matplotlib.pyplot as plt
        resid = predictions - true_values
        plot_acf(resid)
        plt.xlabel('lag')
        plt.ylabel('correlation')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def test_model(self, test_data: pd.DataFrame, horizon: int = 1) -> float:
        """test model on test data, return mape"""
        from sktime.forecasting.model_selection import SlidingWindowSplitter
        from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
        test_target = pd.Series(
            test_data[self.target_col],
            index=pd.to_datetime(test_data[self.date_col])
        )
        test_exog = pd.Series(
            test_data[self.exog_col],
            index=pd.to_datetime(test_data[self.date_col])
        )
        fh = ForecastingHorizon(horizon)
        cv = SlidingWindowSplitter(
            start_with_window=True,
            fh=fh,
            step_length=self.step_length,
            window_length=self.window_length
        )
        results = evaluate(
            forecaster=self.forecaster,
            y=test_target,
            X=test_exog,
            cv=cv,
            scoring=mean_absolute_percentage_error,
            return_data=True,
        ).dropna()
        return results.test__DynamicForecastingErrorMetric.mean()

    def plot_optimal_prices(self, optimal_prices: List[float], historical_prices: pd.Series, dates: pd.Series, save_path: str = None) -> None:
        """plot optimal vs historical prices"""
        import matplotlib.pyplot as plt
        smoothed_optimal = np.convolve(optimal_prices + np.mean(optimal_prices), np.ones(3) / 3, mode='valid')
        plt.plot(dates.iloc[-len(smoothed_optimal):], historical_prices.iloc[-len(smoothed_optimal):], color='orange', label='historical')
        plt.plot(dates.iloc[-len(smoothed_optimal):], smoothed_optimal, color='g', label='optimal')
        plt.ylabel('price')
        plt.legend()
        plt.xticks(rotation=70)
        if save_path:
            plt.savefig(save_path)
        plt.show()




# usage
if __name__ == "__main__":

    data = pd.read_csv("../src/data/california.csv")


    optimizer = TimeSeriesPriceOptimizer(
        data=data,
        target_col="TotalVolume",  # e.g., demand
        exog_col="Delta",         # e.g., price change
        date_col="Date",
        price_col="AveragePrice", # e.g., actual price
        train_size=TimeSeriesPriceOptimizer.TRAIN_SIZE
    )


    best_params = optimizer.train_model()
    print("Best ARIMA Parameters:", best_params)


    predicted_values = optimizer.predict(horizon=5)
    print("Predicted Values:", predicted_values)


    historical_prices = data[optimizer.price_col].iloc[:optimizer.TRAIN_SIZE]
    optimal_prices = optimizer.get_optimal_price(n_periods=20, historical_prices=historical_prices)
    print("Optimal Prices:", optimal_prices)