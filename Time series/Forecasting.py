# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.stattools as sts
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from pmdarima.arima import auto_arima
import yfinance 
import warnings
warnings.filterwarnings('ignore')
# %%
raw_data = yfinance.download(
                    tickers=["ICICIBANK.NS", "^NSEBANK"],
                    start="2014-12-31",
                    end="2021-01-01",
                    group_by="ticker",
                    auto_adjust=True,
                    interval="1d",
                    treads=True
                    )
# %%
df = raw_data.copy()
# %%
df["nsebank"] = df["^NSEBANK"].Close
df["icici"] = df["ICICIBANK.NS"].Close

del  df["^NSEBANK"], df["ICICIBANK.NS"]
df.head()
# %%
df = df.asfreq("b")
df = df.fillna(method="ffill")
# %%
returns =["icici_ret", "nsebank_ret"]
df["nsebank_ret"] = df.nsebank.pct_change(1)*100
df["icici_ret"] = df.icici.pct_change(1)*100
# %%
df.head()
# %%
df_ret = df[returns][1:]
df_ret = df_ret.rename(
                columns={"nsebank_ret": "nsebank",
                "icici_ret": "icici"}
                                )
df_ret.head()
# %%
start_date= "2019-12-31"
end_date="2021-01-01"
X_train = df_ret[ :start_date]
X_test = df_ret[start_date:end_date][1:]
# %%
X_train.tail()
# %%
X_test.head()
# %%
sts.adfuller(X_train.icici)
# %%
sgt.plot_acf(X_train.icici, lags=40, zero=False)
plt.title("ACF ICICI")
sgt.plot_pacf(X_train.icici, lags= 40, zero=False)
plt.title("PACF ICICI")
plt.show()
# %%
model = auto_arima(X_train.icici,
                exogeneous =df_ret[["nsebank"]]
                )
model.summary()
# %%
model1 = arch_model(
            df_ret.icici,
            mean="constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="Normal"
            )
results1 = model1.fit(last_obs= start_date, update_freq=10)
results1.summary()
# %%
pred_garch = results1.forecast(horizon=1, align="target")
pred_garch.residual_variance[start_date:].plot(zorder=2)
X_test.icici.abs().plot(zorder=1)
plt.show()
# %%
pred_garch = results1.forecast(
                        horizon=100,
                        align="target"
)
pred_100 =pred_garch.residual_variance[-30:]
# %%
pred_100.mean().T.plot()

# %%
