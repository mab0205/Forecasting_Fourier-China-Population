# -*- coding: utf-8 -*-
"""
Exemplo de Forecast População China

@author: Martín Ávila 
"""

import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = '/home/mab0205/UTFPR/semestre 2023.2/sistemas inteligentes/Seminario_Martin_Avila/Hoja de cálculo sin título - China population.csv'
data = pd.read_csv(file_path, header=0)

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

print ( data.describe)
# Set 'Date' as index
data.set_index('Date', inplace=True)
data2 = data.copy()
#print(data.columns)

annual_dates = pd.date_range(start='1950-01-01', end='2010-01-01', freq='AS')
annual_dates_full = pd.date_range(start='1950-01-01', end='2022-01-01', freq='AS')

# Reindex DataFrame 
data = data.reindex(annual_dates)
data_full = data2.reindex(annual_dates_full)

# Create deterministic components to capture trend and seasonality
fourier = CalendarFourier(freq='A', order=10)  # 'A' Annual Fourier components

dp = DeterministicProcess(
    index=data.index,
    constant=True,
    order=1,  
    additional_terms=[fourier],
    drop=True,
)

dp_full = DeterministicProcess(
    index=data_full.index,
    constant=True,
    order=1,  
    additional_terms=[fourier],
    drop=True,
)

X = dp.in_sample()  # Independent variables
#X = X[1:]
y = data['Population'].diff().shift(-1).dropna()  # Dependent variable (population)
y_full = data_full['Population']
#y = y[1:] - y[0:-1]

X = X[:-1]

### Normalization ###
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# new Model
model = RandomForestRegressor(n_estimators=100, random_state=3)
model.fit(X_scaled, y)

#  deterministic components, predit 1950 e 2030
future_dates = pd.date_range(start='1950-01-01', end='2030-01-01', freq='AS')
future_dp = DeterministicProcess(
    index=future_dates,
    constant=True,
    order=1,
    additional_terms=[fourier],
    drop=True,
)

# Get the future independent variables and scale them
future_X = future_dp.in_sample()
future_X_scaled = scaler.transform(future_X)

predicted_population_2023 = model.predict(future_X_scaled)

Y = [y[0]]
for val in y[1:]:
    Y.append(val + Y[-1])

YP = predicted_population_2023
yp = [data2['Population'][0]]
for val in YP[1:]:
    yp.append(val + yp[-1])

# Visual
plt.figure(figsize=(10, 6))
plt.plot(data_full.index, data2['Population'], label='Actual Population')
plt.plot(future_dates, yp, label='Predicted Population')
#plt.plot(data.index[:-1], model.predict(X_scaled), label='Fitted Population')
#plt.plot(future_dates, predicted_population_2023, label='Predicted Population 2023', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Population')
plt.title('Actual v Predicted Population')
plt.legend()
plt.show()
