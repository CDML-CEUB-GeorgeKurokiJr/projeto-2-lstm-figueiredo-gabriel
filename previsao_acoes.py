import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURAÇÕES ---
tickers = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'ABEV3.SA', 'BBAS3.SA']
N_PASSOS = 60 
TARGET_SIZE = len(tickers)

# 1. DOWNLOAD E PROCESSAMENTO
def get_data(tickers):
    all_data = []
    for t in tickers:
        print(f"Baixando {t}...")
        df = yf.download(t, start='2019-01-01', end='2026-03-12', progress=False)
        df = df[['Close']].rename(columns={'Close': f'{t}_Close'})
        window = 14
        df[f'{t}_SMA'] = df[f'{t}_Close'].rolling(window=window).mean()
        delta = df[f'{t}_Close'].diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = -delta.clip(upper=0).rolling(window=window).mean()
        df[f'{t}_RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        all_data.append(df)
    return pd.concat(all_data, axis=1).dropna()

df_final = get_data(tickers)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_final)

# 2. JANELAS TEMPORAIS
def create_sequences(data, n_passos, df_ref, tickers_originais):
    X, y = [], []
    indices = [df_ref.columns.get_loc(f'{t}_Close') for t in tickers_originais]
    for i in range(len(data) - n_passos):
        X.append(data[i : i + n_passos])
        # Extração manual para evitar o IndexError do NumPy
        dia_seguinte = data[i + n_passos]
        y.append([dia_seguinte[idx] for idx in indices])
    return np.array(X), np.array(y)

X_np, y_np = create_sequences(scaled_data, N_PASSOS, df_final, tickers)
X_train = torch.tensor(X_np, dtype=torch.float32)
y_train = torch.tensor(y_np, dtype=torch.float32)

# 3. MODELO LSTM CORRIGIDO
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # O .squeeze() garante que o formato seja exatamente [batch, 5]
        return self.linear(hn[-1]).squeeze()

model = StockLSTM(X_train.shape[2], 64, TARGET_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 4. TREINAMENTO
print("\nIniciando treinamento...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    
    # AJUSTE FINAL DE SHAPE PARA O CRITERION
    loss = criterion(outputs.view(y_train.shape), y_train)
    
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Época [{epoch+1}/100], Perda: {loss.item():.6f}')

# 5. GRÁFICO
if not os.path.exists('resultados'): os.makedirs('resultados')
plt.figure(figsize=(12, 6))
plt.plot(y_train[-100:, 0].numpy(), label='Real (VALE3)', color='blue')
plt.plot(outputs.detach()[-100:, 0].numpy(), label='Previsão', color='red', ls='--')
plt.legend()
plt.title('Previsão Multivariada Finalizada')
plt.savefig('resultados/projeto_final.png')
print("\nGráfico salvo em 'resultados/projeto_final.png'")
plt.show()