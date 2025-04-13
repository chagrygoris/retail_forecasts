# test_timeseries_price_optimizer.py
import pytest
import pandas as pd
import numpy as np
from ..price_optimizer import TimeSeriesPriceOptimizer

@pytest.fixture
def synthetic_data():
    return pd.DataFrame({
        "date": pd.date_range(start="2020-01-01", periods=500, freq="D"),
        "value": np.random.normal(1000, 50, 500),
        "exogenous": np.random.normal(0, 0.1, 500),
        "price": np.random.uniform(10, 20, 500)
    })

@pytest.fixture
def alternative_data():
    return pd.DataFrame({
        "timestamp": pd.date_range(start="2019-01-01", periods=600, freq="D"),
        "sales": np.random.normal(500, 30, 600),
        "price_change": np.random.normal(0, 0.05, 600),
        "cost": np.random.uniform(5, 15, 600)
    })

@pytest.fixture(params=[
    ("date", "value", "exogenous", "price"),
    ("timestamp", "sales", "price_change", "cost")
])
def column_names(request):
    return request.param

@pytest.fixture
def optimizer(synthetic_data, alternative_data, column_names, request):
    data_idx = request.node.callspec.params.get("data_idx", 0)
    data = synthetic_data if data_idx == 0 else alternative_data
    date_col, target_col, exog_col, price_col = column_names
    return TimeSeriesPriceOptimizer(
        data=data,
        target_col=target_col,
        exog_col=exog_col,
        date_col=date_col,
        price_col=price_col,
        train_size=400
    )

@pytest.mark.parametrize("data_idx", [0, 1])
def test_init(optimizer, synthetic_data, alternative_data, column_names, data_idx):
    date_col, target_col, exog_col, price_col = column_names
    assert optimizer.target_col == target_col
    assert optimizer.exog_col == exog_col
    assert optimizer.date_col == date_col
    assert optimizer.price_col == price_col
    assert optimizer.train_size == 400
    assert optimizer.forecaster is None

@pytest.mark.parametrize("data_idx", [0, 1])
def test_load_and_prepare_data(optimizer, synthetic_data, alternative_data, column_names, data_idx):
    assert len(optimizer.target_series) == 400
    assert len(optimizer.exog_series) == 400
    assert len(optimizer.price_series) == 400
    assert len(optimizer.dates) == 400
    data = synthetic_data if data_idx == 0 else alternative_data
    date_col = column_names[0]
    assert (optimizer.dates == pd.to_datetime(data[date_col].iloc[:400])).all()

@pytest.mark.parametrize("data_idx", [0, 1])
@pytest.mark.parametrize("max_p, max_q, max_d", [(1, 1, 0), (2, 2, 1)])
def test_train_model(optimizer, max_p, max_q, max_d, data_idx):
    result = optimizer.train_model(max_p=max_p, max_q=max_q, max_d=max_d)
    assert optimizer.forecaster is not None
    assert "order" in result
    assert "score" in result
    assert "params" in result
    assert isinstance(result["order"], tuple)
    assert len(result["order"]) == 3

@pytest.mark.parametrize("data_idx", [0, 1])
@pytest.mark.parametrize("horizon", [3, 5])
def test_predict(optimizer, horizon, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    predictions = optimizer.predict(horizon=horizon)
    assert len(predictions) == horizon
    assert isinstance(predictions, pd.Series)
    # test error when not trained
    new_optimizer = TimeSeriesPriceOptimizer(
        data=synthetic_data if data_idx == 0 else alternative_data,
        target_col=optimizer.target_col,
        exog_col=optimizer.exog_col,
        date_col=optimizer.date_col,
        price_col=optimizer.price_col,
        train_size=400
    )
    with pytest.raises(ValueError, match="train the model first"):
        new_optimizer.predict(horizon=horizon)

@pytest.mark.parametrize("data_idx", [0, 1])
@pytest.mark.parametrize("n_periods", [10, 20])
def test_get_optimal_price(optimizer, synthetic_data, alternative_data, column_names, n_periods, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    data = synthetic_data if data_idx == 0 else alternative_data
    price_col = column_names[3]
    historical_prices = pd.Series(data[price_col].values[:400])
    optimal_prices = optimizer.get_optimal_price(n_periods=n_periods, historical_prices=historical_prices)
    assert len(optimal_prices) == n_periods - 1
    assert all(isinstance(price, (int, float)) for price in optimal_prices)
    # test error when not trained
    new_optimizer = TimeSeriesPriceOptimizer(
        data=data,
        target_col=optimizer.target_col,
        exog_col=optimizer.exog_col,
        date_col=optimizer.date_col,
        price_col=optimizer.price_col,
        train_size=400
    )
    with pytest.raises(ValueError, match="train the model first"):
        new_optimizer.get_optimal_price(n_periods=n_periods, historical_prices=historical_prices)

@pytest.mark.parametrize("data_idx", [0, 1])
@pytest.mark.parametrize("a, intercept, alpha, price_lag_1, x, expected_delta", [
    ([0.1, 0.2], 1.0, -0.5, 10.0, [1.0, 2.0], (-1.0 - np.dot([0.1, 0.2], [1.0, 2.0]) - (-0.5)) / (2 * -0.5))
])
def test_get_optimal_price_linear(optimizer, a, intercept, alpha, price_lag_1, x, expected_delta, data_idx):
    price = optimizer._get_optimal_price_linear(a, intercept, alpha, price_lag_1, x)
    expected_price = price_lag_1 * (1 + expected_delta)
    assert price == pytest.approx(expected_price)

@pytest.mark.parametrize("data_idx", [0, 1])
def test_plot_predictions(optimizer, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    predictions = pd.Series([1000, 1010, 1020, 1030, 1040])
    true_values = pd.Series([1005, 1015, 1025, 1035, 1045])
    optimizer.plot_predictions(predictions, true_values)

@pytest.mark.parametrize("data_idx", [0, 1])
def test_plot_residuals(optimizer, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    predictions = pd.Series([1000, 1010, 1020, 1030, 1040])
    true_values = pd.Series([1005, 1015, 1025, 1035, 1045])
    optimizer.plot_residuals(predictions, true_values)

@pytest.mark.parametrize("data_idx", [0, 1])
@pytest.mark.parametrize("bins", [10, 20])
def test_plot_residual_histogram(optimizer, bins, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    predictions = pd.Series([1000, 1010, 1020, 1030, 1040])
    true_values = pd.Series([1005, 1015, 1025, 1035, 1045])
    optimizer.plot_residual_histogram(predictions, true_values, bins=bins)

@pytest.mark.parametrize("data_idx", [0, 1])
def test_plot_residual_autocorrelation(optimizer, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    predictions = pd.Series([1000, 1010, 1020, 1030, 1040])
    true_values = pd.Series([1005, 1015, 1025, 1035, 1045])
    optimizer.plot_residual_autocorrelation(predictions, true_values)

@pytest.mark.parametrize("data_idx", [0, 1])
@pytest.mark.parametrize("horizon", [1, 2])
def test_test_model(optimizer, synthetic_data, alternative_data, horizon, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    data = synthetic_data if data_idx == 0 else alternative_data
    test_data = data.iloc[400:]
    mape = optimizer.test_model(test_data, horizon=horizon)
    assert isinstance(mape, float)
    assert mape >= 0

@pytest.mark.parametrize("data_idx", [0, 1])
def test_plot_optimal_prices(optimizer, synthetic_data, alternative_data, column_names, data_idx):
    optimizer.train_model(max_p=1, max_q=1, max_d=0)
    data = synthetic_data if data_idx == 0 else alternative_data
    price_col = column_names[3]
    date_col = column_names[0]
    historical_prices = pd.Series(data[price_col].values[:400])
    optimal_prices = optimizer.get_optimal_price(n_periods=20, historical_prices=historical_prices)
    dates = pd.Series(data[date_col].values[:400])
    optimizer.plot_optimal_prices(optimal_prices, historical_prices, dates)