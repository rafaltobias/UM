import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TemperatureRNN:
    def __init__(self, sequence_length=60, rnn_units=50, dropout_rate=0.2):
        self.sequence_length = sequence_length
        self.rnn_units = rnn_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        
    def generate_temperature_data(self, n_days=1000):
        print("Generowanie syntetycznych danych temperatury...")
        
        start_date = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        seasonal_trend = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + 10
        
        weekly_trend = 3 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        
        noise = np.random.normal(0, 2, n_days)
        
        long_term_trend = 0.01 * np.arange(n_days)
        
        temperature = seasonal_trend + weekly_trend + noise + long_term_trend
        
        data = pd.DataFrame({
            'date': start_date,
            'temperature': temperature
        })
        
        return data
    
    def load_data(self, file_path=None):
        if file_path and os.path.exists(file_path):
            print(f"Wczytywanie danych z pliku: {file_path}")
            data = pd.read_csv(file_path)
        else:
            print("Generowanie syntetycznych danych temperatury...")
            data = self.generate_temperature_data()
        
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        print(f"Wczytano {len(data)} obserwacji")
        print(f"Zakres dat: {data['date'].min()} - {data['date'].max()}")
        print(f"Zakres temperatur: {data['temperature'].min():.2f}°C - {data['temperature'].max():.2f}°C")
        
        return data
    
    def prepare_sequences(self, data, target_col='temperature'):
        print(f"Przygotowywanie sekwencji o długości {self.sequence_length}...")
        
        temperature_values = data[target_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(temperature_values)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"Przygotowano {len(X)} sekwencji")
        print(f"Kształt X: {X.shape}, kształt y: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, train_ratio=0.8):
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Podział danych:")
        print(f"Trening: {len(X_train)} próbek")
        print(f"Test: {len(X_test)} próbek")
        
        return X_train, X_test, y_train, y_test

    def build_model(self):
        print("Budowanie modelu RNN...")
        
        model = Sequential()
        model.add(SimpleRNN(self.rnn_units, 
                  return_sequences=True, 
                  activation='tanh',
                  input_shape=(self.sequence_length, 1)))
        model.add(Dropout(self.dropout_rate))
        
        model.add(SimpleRNN(self.rnn_units // 2, 
                  return_sequences=False, 
                  activation='tanh'))
        model.add(Dropout(self.dropout_rate))
        
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(self.dropout_rate / 2))
        model.add(Dense(1, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("Architektura modelu:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        print("Rozpoczęcie treningu modelu...")
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.history = history
        print("Trening zakończony!")
        return history

    def make_predictions(self, X_test):
        print("Wykonywanie predykcji...")
        
        predictions_scaled = self.model.predict(X_test, verbose=0)
        
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def evaluate_model(self, y_true, y_pred):
        print("Ocena jakości modelu...")
        
        y_true_scaled = y_true.reshape(-1, 1)
        y_true_original = self.scaler.inverse_transform(y_true_scaled).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred))
        mae = mean_absolute_error(y_true_original, y_pred)
        mape = np.mean(np.abs((y_true_original - y_pred) / y_true_original)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        print("Metryki oceny:")
        for metric, value in metrics.items():
            if metric == 'MAPE':
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}°C")
        
        return metrics
    
    def plot_training_history(self):
        if self.history is None:
            print("Brak danych o historii treningu!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history.history['loss'], label='Trening', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Walidacja', linewidth=2)
        ax1.set_title('Funkcja straty podczas treningu', fontsize=14)
        ax1.set_xlabel('Epoka')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history.history['mae'], label='Trening', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Walidacja', linewidth=2)
        ax2.set_title('Średni błąd bezwzględny podczas treningu', fontsize=14)
        ax2.set_xlabel('Epoka')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, data, y_test, predictions, n_points=200):
        print("Tworzenie wizualizacji predykcji...")
        
        y_test_scaled = y_test.reshape(-1, 1)
        y_test_original = self.scaler.inverse_transform(y_test_scaled).flatten()
        
        if len(predictions) > n_points:
            step = len(predictions) // n_points
            y_test_plot = y_test_original[::step]
            predictions_plot = predictions[::step]
            indices = np.arange(len(y_test_original))[::step]
        else:
            y_test_plot = y_test_original
            predictions_plot = predictions
            indices = np.arange(len(y_test_original))
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(indices, y_test_plot, label='Rzeczywiste wartości', 
                linewidth=2, alpha=0.8, color='blue')
        plt.plot(indices, predictions_plot, label='Predykcje', 
                linewidth=2, alpha=0.8, color='red')
        plt.title('Porównanie rzeczywistych wartości z predykcjami', fontsize=14)
        plt.xlabel('Indeks próbki')
        plt.ylabel('Temperatura (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        errors = y_test_plot - predictions_plot
        plt.plot(indices, errors, color='green', linewidth=1, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Błędy predykcji', fontsize=14)
        plt.xlabel('Indeks próbki')
        plt.ylabel('Błąd (°C)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Rozkład błędów predykcji', fontsize=14)
        plt.xlabel('Błąd (°C)')
        plt.ylabel('Częstotliwość')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.scatter(y_test_plot, predictions_plot, alpha=0.6, color='orange')
        min_val = min(y_test_plot.min(), predictions_plot.min())
        max_val = max(y_test_plot.max(), predictions_plot.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.title('Rzeczywiste vs Predykcje', fontsize=14)
        plt.xlabel('Rzeczywiste wartości (°C)')
        plt.ylabel('Predykcje (°C)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_temperature_series(self, data, title="Seria czasowa temperatury"):
        plt.figure(figsize=(15, 6))
        plt.plot(data['date'], data['temperature'], linewidth=1, alpha=0.8)
        plt.title(title, fontsize=16)
        plt.xlabel('Data')
        plt.ylabel('Temperatura (°C)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    print("=== ANALIZA DANYCH CZASOWYCH TEMPERATURY Z UŻYCIEM SIECI RNN ===\n")
    
    rnn_model = TemperatureRNN(
        sequence_length=60,
        rnn_units=50,
        dropout_rate=0.2
    )
    
    print("1. WCZYTYWANIE DANYCH")
    print("-" * 50)
    data = rnn_model.load_data()
    
    rnn_model.plot_temperature_series(data, "Oryginalna seria czasowa temperatury")
    
    print("\n2. PRZYGOTOWANIE DANYCH")
    print("-" * 50)
    X, y = rnn_model.prepare_sequences(data)
    X_train, X_test, y_train, y_test = rnn_model.split_data(X, y)
    
    print("\n3. BUDOWANIE MODELU RNN")
    print("-" * 50)
    model = rnn_model.build_model()
    
    print("\n4. TRENING MODELU")
    print("-" * 50)
    history = rnn_model.train_model(
        X_train, y_train, X_test, y_test,
        epochs=100,
        batch_size=32
    )
    
    rnn_model.plot_training_history()
    
    print("\n5. WYKONYWANIE PREDYKCJI")
    print("-" * 50)
    predictions = rnn_model.make_predictions(X_test)
    
    print("\n6. OCENA JAKOŚCI MODELU")
    print("-" * 50)
    metrics = rnn_model.evaluate_model(y_test, predictions)
    
    print("\n7. WIZUALIZACJA WYNIKÓW")
    print("-" * 50)
    rnn_model.plot_predictions(data, y_test, predictions)
    
    print("\n=== ANALIZA ZAKOŃCZONA ===")
    
    return rnn_model, data, predictions, metrics

if __name__ == "__main__":
    model, data, predictions, metrics = main()
    
    print("\nDodatkowe informacje:")
    print(f"Użyto {len(data)} dni danych")
    print(f"Model przewiduje temperaturę na podstawie {model.sequence_length} poprzednich dni")
    print(f"Architektura: RNN z {model.rnn_units} neuronami w pierwszej warstwie")
