# Walmart Sales Prediction

CS1783 Artificial Intelligence Programming
Jacob Shultz, Aidan Gross

This notebook trains an LSTM in PyTorch to predict the next 28 days of total daily sales for one Walmart store (CA_1). It uses the [M5 Forecasting dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and compares the LSTM against two simple baselines.

## Setup

### Data

Download two files from the M5 competition page (needs a Kaggle account):

- `calendar.csv`
- `sales_train_evaluation.csv`

Or use the Kaggle CLI:

```bash
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip calendar.csv sales_train_evaluation.csv
```

The other files are not used. `sales_train_validation.csv` is dropped per the assignment. `sell_prices.csv` and `sample_submission.csv` are not needed.

### Google Colab (as written)

1. Put both CSVs in Google Drive at `MyDrive/Colab Notebooks/datasets/walmart/`.
2. Open `project.ipynb` in Colab.
3. Run all cells. Allow Drive access when asked.

A GPU is optional. The code uses CUDA if it is available.

### Local

Needs Python 3.10+.

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

Replace the `google.colab` cell with `DATASETS = "path/to/walmart/"`.

## Notebook Layout

| Section | Contents |
|---|---|
| 0. Global | Imports, train/test split date, CSV loading |
| 1. Data Processing and Analysis | Daily sales for CA_1, stats, outlier filling, feature encoding |
| 2. LSTM Training Pipeline | `TrainingParameters`, windowing, training loop, experiments |
| 3. Testing | 28-day forecasts vs. actual sales and baselines |

## Data Processing

1. Keep only store `CA_1` (3,049 items).
2. Sum all items for each day. This gives 1,941 days of total unit sales (2011-01-29 to 2016-05-22).
3. Join the daily totals to `calendar.csv`. Drop the 28 calendar days after the sales data ends.
4. Split by date:
   - Train: 2011-01-29 to 2015-04-30 (1,553 days, 80%)
   - Test: 2015-05-01 to 2016-05-22 (388 days)
5. Fill outliers. Days more than 3 standard deviations below the training mean are replaced by linear interpolation. This catches the 5 Christmas days, when the store sold 0 units. The tests in Section 3 score against the original sales, zeros included.
6. Encode features:
   - `wday` and `month` become sine/cosine pairs.
   - `event_type_1` and `event_type_2` are one-hot encoded.
   - `snap_CA` is kept as is.
   - Dropped: `date`, `d`, `wm_yr_wk`, `weekday`, `year`, `event_name_1`, `event_name_2`, `snap_TX`, `snap_WI`.
7. Scale `sales` with `MinMaxScaler`, fit on training days only. The other columns already fall between -1 and 1.

Final input columns (12): `sales`, `snap_CA`, `wday_sin`, `wday_cos`, `month_sin`, `month_cos`, `event_type_1_Cultural`, `event_type_1_National`, `event_type_1_Religious`, `event_type_1_Sporting`, `event_type_2_Cultural`, `event_type_2_Religious`.

## Model

```
input (90 days x 12 columns)
  -> LSTM (hidden_size units, lstm_layers layers)
  -> output at the last time step
  -> Linear -> 28 predicted days
```

The model predicts all 28 days at once.

Each window is sorted by its 28 target days. If all 28 are training days, it is a training window. If all 28 are test days, it is a test window. Windows that cross the split are dropped. With a 90-day input, this gives 1,436 training windows and 361 test windows.

Training uses shuffled batches. After each epoch, the model is scored on all test windows. "Real MAE" is the mean absolute error in units per day, after undoing the scaling. Each run plots train loss, test loss, and real MAE by epoch.

Each run is set by a `TrainingParameters` dataclass:

| Parameter | Default |
|---|---|
| `features` | required |
| `sequence_length` | 90 |
| `prediction_length` | 28 |
| `hidden_size` | 32 |
| `lstm_layers` | 1 |
| `dropout` | 0.0 (only used when `lstm_layers > 1`) |
| `batch_size` | 32 |
| `learning_rate` | 1e-3 |
| `epochs` | 50 |
| `optimizer` | `torch.optim.Adam` |
| `loss_function` | `nn.L1Loss` (MAE) |
| `seed` | `None` |

To add an experiment, add a `TrainingParameters(...)` entry to `p_list`.

## Results

Numbers are from the saved notebook outputs. The experiments do not set `seed`, so reruns will differ a little. All MAE values are units per day. Mean daily sales in the test period is about 4,470.

### Experiments

All configs use 90 input days, all 12 columns, 1 LSTM layer, no dropout, batch size 32, and Adam. The first row is the assignment's starting config.

| Hidden size | Learning rate | Epochs | Loss | Real MAE |
|---|---|---|---|---|
| 32 | 1e-3 | 50 | MSE | 308.9 |
| 64 | 1e-3 | 50 | MAE | 290.5 |
| 128 | 5e-4 | 100 | MAE | 282.2 |
| 128 | 1e-3 | 100 | MAE | 287.7 |
| 128 | 5e-4 | 100 | MSE | 277.1 |
| **128** | **1e-3** | **100** | **MSE** | **275.4** |

Real MAE is from the last epoch. The bold row has the lowest MAE and is used for all tests. The experiment cell also has commented-out configs for 2 LSTM layers with dropout, 56- and 364-day inputs, and Huber loss.

### Tests

Each test picks a "today" and gives the model the 90 days ending on that day. The model predicts the next 28 days. The forecast is compared to actual sales and two baselines:

- **Same day last month:** actual sales from 28 days earlier.
- **Same day last year:** actual sales from 364 days earlier.

Both baselines shift by whole weeks, so the day of the week lines up.

**Test 1: start of the test period.** Today is 2015-04-30. Predicts 2015-05-01 to 2015-05-28.

| LSTM | Same day last month | Same day last year |
|---|---|---|
| 224.2 | 380.9 | 366.2 |

**Test 2: 10 forecasts inside the test period.**

| Today | LSTM | Same day last month | Same day last year |
|---|---|---|---|
| 2015-05-15 | 194.8 | 299.9 | 299.8 |
| 2015-06-19 | 341.2 | 427.2 | 373.2 |
| 2015-07-24 | 187.4 | 470.1 | 170.2 |
| 2015-08-28 | 284.7 | 368.9 | 372.3 |
| 2015-10-02 | 185.2 | 324.5 | 294.5 |
| 2015-11-06 | 290.8 | 496.1 | 307.9 |
| 2015-12-11 | 344.9 | 428.0 | 584.1 |
| 2016-01-15 | 321.4 | 649.7 | 317.7 |
| 2016-03-04 | 335.1 | 478.8 | 447.8 |
| 2016-04-22 | 323.5 | 409.3 | 329.7 |
| **Average** | **280.9** | **435.3** | **349.7** |

The LSTM has the lowest MAE in 8 of 10 forecasts. "Same day last year" wins on 2015-07-24 and 2016-01-15.