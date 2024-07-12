import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, \
    mean_absolute_error
from sklearn.model_selection import train_test_split
from mapie.subsample import Subsample
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
from mapie.regression import MapieRegressor
from xgboost import XGBRegressor
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str,
                    help="Path to training data csv.")
parser.add_argument("--path_model", type=str,
                    help="Path to directory where the trained models" +
                    " are stored.")
parser.add_argument("--model_name", type=str,
                    help="Name of the model iteration.")
parser.add_argument("--scaler_feats", type=str,
                    help="Path to the feature scaler.")
args = parser.parse_args()


warnings.filterwarnings("ignore")

path_dat = args.path_data
path_results = args.path_model
if not os.path.isdir(path_results):
    os.mkdir(path_results)
model_prefix = "MAPIE_"
path_results = f"{args.path_model}/{args.model_name}"
if not os.path.isdir(path_results):
    os.mkdir(path_results)


# =============================================================== #
def yerr(y_pred, intervals):
    """
    Returns the error bars with the point prediction and its interval

    Parameters
    ----------
    y_pred: ArrayLike
        Point predictions.
    intervals: ArrayLike
        Predictions intervals.

    Returns
    -------
    ArrayLike
        Error bars.
    """
    return np.abs(np.concatenate(
        [
            np.expand_dims(y_pred, 0) - intervals[:, 0, 0].T,
            intervals[:, 1, 0].T - np.expand_dims(y_pred, 0),
        ],
        axis=0,
    ))


# =============================================================== #
df_final = pd.read_csv(path_dat)
df_final = df_final.loc[(df_final.particle_number <= 5e4) &
                        (df_final.no2 != -999.0) &
                        (df_final.pm25 > 0) &
                        (df_final["pop"] > 0) &
                        (df_final.particle_number > 150) &
                        ((df_final.particle_number != 1545.0))]
df_final = df_final.dropna()


# ====================== MACHINE LEARNING ================== #
feats = ['land_1', 'land_2', 'land_3', 'land_4', 'land_5',
         'land_6', 'land_7', 'pop', 'buildUp', 'degreeUrb', 'humanSettle',
         'pm25', "no2", 'blackCarbon', 'carbonDioxide', 'carbonMonoxide',
         'nitrogenOxides', "t2m"]

scaler_feats = joblib.load(
    args.scaler_feats)

target = "particle_number"

land_maps = {"land_1": "Forest", "land_2": "Low Vegetation",
             "land_3": "Inland Water", "land_4": "Cropland",
             "land_5": "Urban", "land_6": "Snow/Ice",
             "land_7": "Open Sea"}

gc.collect()

# ---------------------- Conformal Predictions ----------------------- #
# Subset and scale the input variables
scaler_target = MinMaxScaler()

X = scaler_feats.transform(df_final[feats])
y = scaler_target.fit_transform(df_final[target].values.reshape(-1, 1))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1,
                                                  shuffle=True)


# Parameter set for model
params_ = dict(n_estimators=200,
               num_parallel_tree=1,
               max_depth=5,
               tree_method="approx",
               eta=0.075,
               subsample=0.75,
               colsample_bytree=0.7,
               n_jobs=10,
               enable_categorical=False)
STRATEGIES = {
    "naive": dict(method="naive"),
    "jackknife": dict(method="base", cv=-1),
    "jackknife_plus": dict(method="plus", cv=-1),
    "jackknife_minmax": dict(method="minmax", cv=-1),
    "cv": dict(method="base", cv=10),
    "cv_plus": dict(method="plus", cv=10),
    "cv_minmax": dict(method="minmax", cv=10),
    "jackknife_plus_ab": dict(method="plus",
                              cv=Subsample(n_resamplings=20)),
    "jackknife_minmax_ab": dict(
        method="minmax", cv=Subsample(n_resamplings=5))}

paramsMAPIE = dict(**STRATEGIES["jackknife_plus_ab"],
                   n_jobs=-1, agg_function="median",
                   verbose=5)
model = MapieRegressor(XGBRegressor(**params_),
                       **paramsMAPIE)

# Fit the model
model.fit(X_train, y_train.ravel())


# Calculate the metrics on the evaluation set (prediction and interval)
pred, int_ = model.predict(X_val, alpha=[0.05])

pred_ = scaler_target.inverse_transform(pred.reshape(-1, 1))
# Calculate the error bars
y_err = np.mean(yerr(pred, int_), axis=0)
errors_ = scaler_target.inverse_transform(y_err.reshape(-1, 1))
# Scale the validation set back
y_val_ = scaler_target.inverse_transform(y_val.reshape(-1, 1))
print(f"MAE: {mean_absolute_error(y_val_, pred_)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val_, pred_))}")
print(f"r2 score: {r2_score(y_val_, pred_)}")

df_val = pd.DataFrame({"gt": y_val_.ravel(),
                       "pred": pred_.ravel(),
                       "error": errors_.ravel()})

sns.set(font_scale=2)
sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.errorbar(data=df_val, x="gt", y="pred", yerr="error", marker="o",
            linestyle="", alpha=0.4)
sns.scatterplot(data=df_val, x="gt", y="pred",
                color="black", ax=ax, s=320)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([np.min(y_val_), np.max(y_val_)])
ax.set_ylim([np.min(y_val_), np.max(y_val_)])
diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="black",
                     linewidth=6, alpha=0.5)
ax.set(xlabel="Ground Truth (cm$^{-3}$)",
       ylabel="Model Prediction (cm$^{-3}$)")
ax.legend("", frameon=False)
plt.tight_layout()
plt.savefig(f"{path_results}/{model_prefix}_EVAL.png")
plt.close()


# ------ Now fit on all the data ------ #
model = MapieRegressor(XGBRegressor(**params_),
                       **paramsMAPIE)
model.fit(X, y.ravel())

# Calculate the metrics on the evaluation set (prediction and interval)
pred, int_ = model.predict(X, alpha=[0.1])
# Calculate the error bars
y_err = np.mean(yerr(pred, int_), axis=0)

pred_ = scaler_target.inverse_transform(pred.reshape(-1, 1))
# Split the intervals into low and upper intervals
errors_ = scaler_target.inverse_transform(y_err.reshape(-1, 1))
# Scale the validation set back
y_ = scaler_target.inverse_transform(y.reshape(-1, 1))
print(f"MAE: {mean_absolute_error(y_, pred_)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_, pred_))}")
print(f"r2 score: {r2_score(y_, pred_)}")


# coverage = regression_coverage_score(y_, y_err[:, 0, 0], y_err[:, 1, 0])
# print(f"Coverage: {coverage:.3f}")

df_train = pd.DataFrame({"gt": y_.ravel(),
                         "pred": pred_.ravel(),
                         "error": errors_.ravel()})

sns.set(font_scale=2)
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.errorbar(data=df_train, x="gt",
            y="pred", yerr="error",
            fmt=".", linewidth=3, alpha=0.15)
sns.scatterplot(data=df_train, color="blue",
                x="gt", y="pred",
                ax=ax, s=320)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([df_train.pred.min(), df_train.pred.max()])
ax.set_ylim([df_train.pred.min(), df_train.pred.max()])
diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="black",
                     linewidth=6, alpha=0.5)
ax.set(xlabel="Ground Truth (cm$^{-3}$)",
       ylabel="Model Prediction (cm$^{-3}$)")
ax.legend("", frameon=False)
plt.tight_layout()
plt.savefig(f"{path_results}/{model_prefix}_TRAINING.png")
plt.close()


# Plot both on the same axis
df_res = pd.concat([df_train.assign(data="Training"),
                    df_val.assign(data="Validation")])

sns.set(font_scale=2.5)
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.errorbar(data=df_res, x="gt",
            y="pred", yerr="error",
            fmt=".", linewidth=3, alpha=0.15)
sns.scatterplot(data=df_res, color="blue",
                x="gt", y="pred", palette="bright",
                ax=ax, s=320, hue="data")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([df_res.pred.min(), df_res.pred.max()])
ax.set_ylim([df_res.pred.min(), df_res.pred.max()])
diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="black",
                     linewidth=6, alpha=0.5)
ax.set(xlabel="Ground Truth (cm$^{-3}$)",
       ylabel="Model Prediction (cm$^{-3}$)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{path_results}/{model_prefix}_BOTH.png")
plt.close()

# ==================================================================== #
df_res.to_parquet(f"{path_results}/{model_prefix}_data.parquet",
                  index=False)
joblib.dump(scaler_target,
            f"{path_results}/{model_prefix}_target_scaler.joblib")
joblib.dump(model, f"{path_results}/{model_prefix}.joblib")
# ==================================================================== #
