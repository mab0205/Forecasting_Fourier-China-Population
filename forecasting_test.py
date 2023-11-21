# -*- coding: utf-8 -*-
"""
Exemplo de Forecast População China

@author: Martín Ávila 
"""

import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

file_path = '/home/mab0205/UTFPR/semestre 2023.2/sistemas inteligentes/Seminario_Martin_Avila/Hoja de cálculo sin título - China population.csv'
data = pd.read_csv(file_path, header=0)

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

print ( data.describe)
# Set 'Date' as index
data.set_index('Date', inplace=True)
#print(data.columns)

annual_dates = pd.date_range(start='1950-01-01', periods=73, freq='AS')

# Reindex DataFrame 
data = data.reindex(annual_dates)

# Create deterministic components to capture trend and seasonality
fourier = CalendarFourier(freq='A', order=10)  # 'A' Annual Fourier components

dp = DeterministicProcess(
    index=data.index,
    constant=True,
    order=2,  
    additional_terms=[fourier],
    drop=True,
)

# Ajustar el modelo de regresión lineal
model = LinearRegression()
X = dp.in_sample()  # Independent variables
y = data['Population']  # Dependent variable (population)

model.fit(X, y)

predicted_population = model.predict(X)

# Visuali
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Population'], label='Actual Population')
plt.plot(data.index, predicted_population, label='Predicted Population')
plt.xlabel('Data')
plt.ylabel('Population')
plt.title('Actual v Predicted Population')
plt.legend()
plt.show()
