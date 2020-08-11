# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing #
import statsmodels.graphics.tsaplots as tsa_plots


# print dataset
dataset = pd.read_csv("/Users/Ankan/PycharmProjects/Basiccode/1. Assignments/15. Forecasting - Done/Airlines Data.csv")
# split dataseet into time, data, year
dataset.index = pd.to_datetime(dataset.Month,format="%b-%y")
print(dataset.columns)
dataset1=dataset["Passengers"]
dataset1.plot()
plt.show()

dataset["Date"] = pd.to_datetime(dataset.Month,format="%b-%y")
dataset["month"] = dataset.Date.dt.strftime("%b")
dataset["year"] = dataset.Date.dt.strftime("%Y")
dataset.month = (pd.to_datetime(dataset.month, format='%b'))

# plot heatmap
heatmap_y_month = pd.pivot_table(data=dataset,values="Passengers",index=["year"],columns=['month'],aggfunc="mean",fill_value=0)
heatmap_y_month.columns = heatmap_y_month.columns.strftime('%b')
sns.heatmap(heatmap_y_month,annot=True,fmt="g")
plt.show()

# plot boxplot
sns.boxplot(x="month",y="Passengers",data=dataset)
plt.show()
sns.boxplot(x="year",y="Passengers",data=dataset)
plt.show()

#plot lineplot
sns.lineplot(x="year",y="Passengers",hue="month",data=dataset)
plt.show()

# moving average for the time series to understand better about the trend character in dataset
dataset1= dataset["Passengers"]
dataset1.plot()
dataset1.plot(label="org")
for i in range(2,24,6):
    dataset["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
plt.show()
# Time series decomposition plot
decompose_ts_add = seasonal_decompose(dataset["Passengers"],model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(dataset["Passengers"],model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets
tsa_plots.plot_acf(dataset["Passengers"],lags=10)
plt.show()
tsa_plots.plot_pacf(dataset["Passengers"])
plt.show()

#split dataset into train & test
Train = dataset.head(80)
Test = dataset.tail(16)
# Creating a function to calculate the MAPE value for test data
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)
# Holt method
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)
# Simple Exponential Method   #seasonal = add
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers)
# Simple Exponential Method   #seasonal = mul
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)

# Lets us use auto_arima from p
auto_arima_model = auto_arima(Train["Passengers"], start_p=0,
                              start_q=0, max_p=10, max_q=10,
                              m=12, start_P=0, seasonal=True,
                              d=1, D=1, trace=True, error_action="ignore",
                              suppress_warnings=True,
                              stepwise=False)

# Visualization of Forecasted values for Test data set using different methods
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.xlabel("Passengers Train")
plt.show()
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.xlabel("Passengers Test")
plt.show()
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.xlabel('Holt Winter,Exponential_1,Exponential_2 ')
plt.show()
