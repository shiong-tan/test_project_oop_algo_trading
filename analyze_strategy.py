import pandas as pd
import numpy as np

# Load and examine the data
url = 'http://hilpisch.com/ref_eikon_eod_data.csv'
symbol = 'AAPL.O'
raw = pd.read_csv(url, index_col=0, parse_dates=True)
data = pd.DataFrame(raw[symbol].iloc[:1000])

# Basic statistics
print('='*80)
print('DATA OVERVIEW')
print('='*80)
print(f'Date range: {data.index[0]} to {data.index[-1]}')
print(f'Total observations: {len(data)}')
print(f'Price range: ${data[symbol].min():.2f} - ${data[symbol].max():.2f}')
print(f'Mean price: ${data[symbol].mean():.2f}')
print(f'Std dev: ${data[symbol].std():.2f}')

# Calculate returns
data['r'] = np.log(data[symbol] / data[symbol].shift(1))
data.dropna(inplace=True)

print('\n' + '='*80)
print('RETURN STATISTICS')
print('='*80)
print(f'Mean daily return: {data["r"].mean()*100:.4f}%')
print(f'Std daily return: {data["r"].std()*100:.4f}%')
print(f'Annualized return: {data["r"].mean()*252*100:.2f}%')
print(f'Annualized volatility: {data["r"].std()*np.sqrt(252)*100:.2f}%')
print(f'Min return: {data["r"].min()*100:.2f}%')
print(f'Max return: {data["r"].max()*100:.2f}%')

# Direction distribution
data['d'] = np.where(data['r'] > 0, 1, 0)
print('\n' + '='*80)
print('DIRECTION DISTRIBUTION (TARGET VARIABLE)')
print('='*80)
print(data['d'].value_counts(normalize=True))
print(f'Up days: {(data["d"] == 1).sum()} ({(data["d"] == 1).sum()/len(data)*100:.1f}%)')
print(f'Down days: {(data["d"] == 0).sum()} ({(data["d"] == 0).sum()/len(data)*100:.1f}%)')

# Buy and hold performance
buy_hold_return = (data[symbol].iloc[-1] / data[symbol].iloc[0] - 1) * 100
print('\n' + '='*80)
print('BUY & HOLD BENCHMARK')
print('='*80)
print(f'Total return: {buy_hold_return:.2f}%')
print(f'Annualized return: {(((data[symbol].iloc[-1] / data[symbol].iloc[0]) ** (252/len(data))) - 1) * 100:.2f}%')
