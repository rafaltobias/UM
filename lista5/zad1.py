import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import cv2
from PIL import Image
import requests
from io import BytesIO
import zipfile
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CNNImageClassifier:
    def __init__(self, model_name='ResNet50', input_shape=(224, 224, 3), num_classes=10):
        """
        Inicjalizacja klasyfikatora obrazów CNN z transfer learning
        
        Args:
            model_name (str): Nazwa predefiniowanej sieci ('ResNet50', 'VGG16', 'MobileNetV2')
            input_shape (tuple): Kształt obrazów wejściowych
            num_classes (int): Liczba klas do klasyfikacji
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.base_model = None
        self.history = None
        self.class_names = None
        self.label_encoder = LabelEncoder()
        

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU dostępne - używanie GPU do treningu")
        else:
            print("GPU niedostępne - używanie CPU")
    
    def download_sample_dataset(self, dataset_type='cifar10'):
        """
        Pobiera przykładowy zbiór danych do testowania
        
        Args:
            dataset_type (str): Typ zbioru danych ('cifar10', 'cifar100')
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, class_names)
        """
        print(f"Pobieranie zbioru danych {dataset_type}...")
        
        if dataset_type == 'cifar10':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
            self.num_classes = 10
            
        elif dataset_type == 'cifar100':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
            class_names = [f'class_{i}' for i in range(100)]
            self.num_classes = 100
            
        else:
            raise ValueError("Nieobsługiwany typ zbioru danych")
        
        self.class_names = class_names
        
        print(f"Pobrano dane:")
        print(f"Trening: {X_train.shape[0]} obrazów")
        print(f"Test: {X_test.shape[0]} obrazów")
        print(f"Rozmiar obrazu: {X_train.shape[1:3]}")
        print(f"Liczba klas: {len(class_names)}")
        
        return X_train, y_train, X_test, y_test, class_names
    
    def preprocess_data(self, X_train, y_train, X_test, y_test):
        """
        Preprocessuje dane obrazowe
        
        Args:
            X_train, y_train: Dane treningowe
            X_test, y_test: Dane testowe
            
        Returns:
            tuple: Preprocessowane dane
        """
        print("Preprocessowanie danych...")
        
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        if X_train.shape[1:3] != self.input_shape[:2]:
            print(f"Zmiana rozmiaru obrazów z {X_train.shape[1:3]} na {self.input_shape[:2]}")
            
            input_shape_smaller = (96, 96, 3) 
            print(f"Używanie mniejszego rozmiaru wejściowego: {input_shape_smaller[:2]} zamiast {self.input_shape[:2]}")
            self.input_shape = input_shape_smaller
            
            batch_size = 1000
            
            X_train_resized = np.zeros((X_train.shape[0], *self.input_shape), dtype='float32')
            X_test_resized = np.zeros((X_test.shape[0], *self.input_shape), dtype='float32')
            
            print("Zmiana rozmiaru obrazów treningowych (w partiach)...")
            num_batches = int(np.ceil(X_train.shape[0] / batch_size))
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, X_train.shape[0])
                for i in range(start_idx, end_idx):
                    X_train_resized[i] = cv2.resize(X_train[i], self.input_shape[:2])
                print(f"\rPostęp: {end_idx}/{X_train.shape[0]} obrazów", end="")
            print() 
            
            print("Zmiana rozmiaru obrazów testowych (w partiach)...")
            num_batches = int(np.ceil(X_test.shape[0] / batch_size))
            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, X_test.shape[0])
                for i in range(start_idx, end_idx):
                    X_test_resized[i] = cv2.resize(X_test[i], self.input_shape[:2])
                print(f"\rPostęp: {end_idx}/{X_test.shape[0]} obrazów", end="")
            print()  # Nowa linia po zakończeniu
                
            X_train, X_test = X_train_resized, X_test_resized
        
        # Konwersja etykiet do one-hot encoding
        y_train_categorical = to_categorical(y_train, self.num_classes)
        y_test_categorical = to_categorical(y_test, self.num_classes)
        
        print(f"Kształt danych po preprocessingu:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train_categorical.shape}")
        print(f"y_test: {y_test_categorical.shape}")
        
        return X_train, y_train_categorical, X_test, y_test_categorical
    
    def create_data_generators(self, X_train, y_train, batch_size=32, validation_split=0.2):
        """
        Tworzy generatory danych z augmentacją
        
        Args:
            X_train, y_train: Dane treningowe
            batch_size (int): Rozmiar batcha
            validation_split (float): Proporcja danych walidacyjnych
            
        Returns:
            tuple: (train_generator, validation_generator)
        """
        print("Tworzenie generatorów danych z augmentacją...")
        
        # Generator z augmentacją dla danych treningowych
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Generator bez augmentacji dla danych walidacyjnych
        validation_datagen = ImageDataGenerator(validation_split=validation_split)
          # Tworzenie generatorów
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            subset='training'
        )
        
        validation_generator = validation_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            subset='validation'
        )
        
        # Obliczenie liczby próbek na podstawie indeksów
        if hasattr(train_generator, 'samples'):
            train_samples = train_generator.samples
            val_samples = validation_generator.samples
        else:
            # Dla nowszych wersji TensorFlow/Keras
            train_samples = len(train_generator.x)
            val_samples = len(validation_generator.x)
        
        print(f"Generator treningowy: {train_samples} próbek")
        print(f"Generator walidacyjny: {val_samples} próbek")
        
        return train_generator, validation_generator
    
    def build_model(self, fine_tune_layers=0):
        """
        Buduje model CNN z transfer learning
        
        Args:
            fine_tune_layers (int): Liczba warstw do fine-tuningu (0 = tylko top layers)
            
        Returns:
            tensorflow.keras.Model: Skompilowany model
        """
        print(f"Budowanie modelu {self.model_name} z transfer learning...")
        
        # Wybór predefiniowanej sieci
        if self.model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Nieobsługiwany model: {self.model_name}")
        
        self.base_model = base_model
        
        # Zamrożenie warstw bazowego modelu
        if fine_tune_layers == 0:
            base_model.trainable = False
            print("Wszystkie warstwy bazowego modelu zamrożone")
        else:
            base_model.trainable = True
            # Zamrożenie pierwszych warstw, fine-tuning ostatnich
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
            print(f"Fine-tuning ostatnich {fine_tune_layers} warstw")
        
        # Dodanie custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Tworzenie finalnego modelu
        model = Model(inputs=base_model.input, outputs=predictions)
          # Kompilacja modelu
        if fine_tune_layers > 0:
            # Niższy learning rate dla fine-tuningu
            learning_rate = 1e-5
        else:
            learning_rate = 1e-3
            
        # Importuj TopKCategoricalAccuracy dla metryki top-5
        from tensorflow.keras.metrics import TopKCategoricalAccuracy
        top5_acc = TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
            
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', top5_acc]
        )
        
        print("Architektura modelu:")
        model.summary()
        
        print(f"Liczba parametrów trenowalnych: {model.count_params()}")
        print(f"Liczba warstw trenowalnych: {sum([layer.trainable for layer in model.layers])}")
        
        self.model = model
        return model
    
    def train_model(self, train_generator, validation_generator, epochs=50):
        """
        Trenuje model
        
        Args:
            train_generator: Generator danych treningowych
            validation_generator: Generator danych walidacyjnych
            epochs (int): Liczba epok
            
        Returns:
            tensorflow.keras.callbacks.History: Historia treningu
        """
        print("Rozpoczęcie treningu modelu...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            f'best_{self.model_name.lower()}_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Trening
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        self.history = history
        print("Trening zakończony!")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Ocenia model na danych testowych
        
        Args:
            X_test: Dane testowe
            y_test: Etykiety testowe
            
        Returns:
            dict: Metryki oceny modelu
        """
        print("Ocena modelu na danych testowych...")
        
        # Predykcje
        predictions = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Obliczanie metryk
        accuracy = accuracy_score(y_true, y_pred)
        
        # Top-5 accuracy (jeśli mamy więcej niż 5 klas)
        if self.num_classes >= 5:
            top5_predictions = np.argsort(predictions, axis=1)[:, -5:]
            top5_accuracy = np.mean([y_true[i] in top5_predictions[i] for i in range(len(y_true))])
        else:
            top5_accuracy = accuracy
        
        metrics = {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true
        }
        
        print(f"Dokładność (Accuracy): {accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """
        Wizualizuje historię treningu
        """
        if self.history is None:
            print("Brak danych o historii treningu!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Trening', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Walidacja', linewidth=2)
        axes[0, 0].set_title('Dokładność podczas treningu', fontsize=14)
        axes[0, 0].set_xlabel('Epoka')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Trening', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Walidacja', linewidth=2)
        axes[0, 1].set_title('Funkcja straty podczas treningu', fontsize=14)
        axes[0, 1].set_xlabel('Epoka')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy (jeśli dostępne)
        if 'top_5_accuracy' in self.history.history:
            axes[1, 0].plot(self.history.history['top_5_accuracy'], label='Trening', linewidth=2)
            axes[1, 0].plot(self.history.history['val_top_5_accuracy'], label='Walidacja', linewidth=2)
            axes[1, 0].set_title('Top-5 Accuracy podczas treningu', fontsize=14)
            axes[1, 0].set_xlabel('Epoka')
            axes[1, 0].set_ylabel('Top-5 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], linewidth=2, color='red')
            axes[1, 1].set_title('Learning Rate podczas treningu', fontsize=14)
            axes[1, 1].set_xlabel('Epoka')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Tworzy macierz pomyłek
        
        Args:
            y_true: Prawdziwe etykiety
            y_pred: Predykcje modelu
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        
        if self.num_classes <= 20:  
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names[:self.num_classes],
                       yticklabels=self.class_names[:self.num_classes])
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        else:
            sns.heatmap(cm, cmap='Blues')
        
        plt.title(f'Macierz pomyłek - {self.model_name}', fontsize=16)
        plt.xlabel('Predykcje')
        plt.ylabel('Prawdziwe etykiety')
        plt.tight_layout()
        plt.show()
    
    def plot_sample_predictions(self, X_test, y_true, y_pred, predictions, num_samples=16):
        """
        Wizualizuje przykładowe predykcje
        
        Args:
            X_test: Dane testowe
            y_true: Prawdziwe etykiety
            y_pred: Predykcje modelu
            predictions: Prawdopodobieństwa predykcji
            num_samples: Liczba próbek do wyświetlenia
        """
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        # Wybór losowych próbek
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            # Wyświetlenie obrazu
            img = X_test[idx]
            if img.max() <= 1.0:  # Jeśli znormalizowane
                img = (img * 255).astype(np.uint8)
            
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Prawdziwa klasa i predykcja
            true_class = self.class_names[y_true[idx]] if self.class_names else f"Class {y_true[idx]}"
            pred_class = self.class_names[y_pred[idx]] if self.class_names else f"Class {y_pred[idx]}"
            confidence = predictions[idx][y_pred[idx]]
            
            # Kolor tytułu - zielony dla poprawnych, czerwony dla błędnych
            color = 'green' if y_true[idx] == y_pred[idx] else 'red'
            
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}',
                             fontsize=10, color=color)
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_accuracy(self, y_true, y_pred):
        """
        Wizualizuje dokładność dla każdej klasy
        
        Args:
            y_true: Prawdziwe etykiety
            y_pred: Predykcje modelu
        """
        if self.num_classes > 20:
            print("Zbyt wiele klas do wizualizacji dokładności per klasa")
            return
            
        # Obliczanie dokładności per klasa
        class_accuracies = []
        for i in range(self.num_classes):
            mask = y_true == i
            if mask.sum() > 0:
                accuracy = (y_pred[mask] == i).mean()
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(self.num_classes), class_accuracies, alpha=0.7)
        plt.title('Dokładność dla każdej klasy', fontsize=16)
        plt.xlabel('Klasa')
        plt.ylabel('Dokładność')
        plt.xticks(range(self.num_classes), 
                  [self.class_names[i] if self.class_names else f'Class {i}' 
                   for i in range(self.num_classes)], 
                  rotation=45)
        
        # Dodanie wartości na słupkach
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Generuje szczegółowy raport klasyfikacji
        
        Args:
            y_true: Prawdziwe etykiety
            y_pred: Predykcje modelu
        """
        if self.class_names and len(self.class_names) >= self.num_classes:
            target_names = self.class_names[:self.num_classes]
        else:
            target_names = [f'Class {i}' for i in range(self.num_classes)]
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("Szczegółowy raport klasyfikacji:")
        print("=" * 60)
        print(report)
        
        return report

def main():
    """
    Główna funkcja wykonująca pełny pipeline klasyfikacji obrazów CNN
    """
    print("=== KLASYFIKACJA OBRAZÓW Z UŻYCIEM KONWOLUCYJNYCH SIECI NEURONOWYCH ===\n")
    
    # Parametry
    MODEL_NAME = 'ResNet50'  # Można zmienić na 'VGG16' lub 'MobileNetV2'
    DATASET = 'cifar10'      # Można zmienić na 'cifar100'
    EPOCHS = 5
    BATCH_SIZE = 32
    
    print(f"Konfiguracja:")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET}")
    print(f"Epoki: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)
      # 1. Inicjalizacja modelu
    classifier = CNNImageClassifier(
        model_name=MODEL_NAME,
        input_shape=(96, 96, 3)  # Zmniejszony rozmiar wejściowy dla oszczędności pamięci
    )
    
    # 2. Pobieranie i przygotowanie danych
    print("\n1. POBIERANIE I PRZYGOTOWANIE DANYCH")
    print("-" * 60)
    X_train, y_train, X_test, y_test, class_names = classifier.download_sample_dataset(DATASET)
    X_train, y_train, X_test, y_test = classifier.preprocess_data(X_train, y_train, X_test, y_test)
    
    # 3. Tworzenie generatorów danych
    print("\n2. TWORZENIE GENERATORÓW DANYCH")
    print("-" * 60)
    train_generator, validation_generator = classifier.create_data_generators(
        X_train, y_train, batch_size=BATCH_SIZE
    )
    
    # 4. Budowanie modelu
    print("\n3. BUDOWANIE MODELU CNN")
    print("-" * 60)
    model = classifier.build_model(fine_tune_layers=0)  # Można zmienić na >0 dla fine-tuningu
    
    # 5. Trening modelu
    print("\n4. TRENING MODELU")
    print("-" * 60)
    history = classifier.train_model(train_generator, validation_generator, epochs=EPOCHS)
    
    # Wizualizacja historii treningu
    classifier.plot_training_history()
    
    # 6. Ocena modelu
    print("\n5. OCENA MODELU NA DANYCH TESTOWYCH")
    print("-" * 60)
    metrics = classifier.evaluate_model(X_test, y_test)
    
    # 7. Szczegółowa analiza wyników
    print("\n6. SZCZEGÓŁOWA ANALIZA WYNIKÓW")
    print("-" * 60)
    
    # Raport klasyfikacji
    classifier.generate_classification_report(metrics['y_true'], metrics['y_pred'])
    
    # Wizualizacje
    print("\n7. WIZUALIZACJE")
    print("-" * 60)
    
    # Macierz pomyłek
    classifier.plot_confusion_matrix(metrics['y_true'], metrics['y_pred'])
    
    # Dokładność per klasa
    classifier.plot_class_accuracy(metrics['y_true'], metrics['y_pred'])
    
    # Przykładowe predykcje
    classifier.plot_sample_predictions(
        X_test, metrics['y_true'], metrics['y_pred'], metrics['predictions']
    )
    
    print("\n=== ANALIZA ZAKOŃCZONA ===")
    print(f"Finalna dokładność: {metrics['accuracy']:.4f}")
    print(f"Top-5 dokładność: {metrics['top5_accuracy']:.4f}")
    
    return classifier, metrics

def compare_models():
    """
    Funkcja porównująca różne architektury CNN
    """
    print("=== PORÓWNANIE RÓŻNYCH ARCHITEKTUR CNN ===\n")
    
    models_to_compare = ['ResNet50', 'VGG16', 'MobileNetV2']
    results = {}
    
    for model_name in models_to_compare:
        print(f"\nTestowanie modelu: {model_name}")
        print("-" * 50)
        
        # Inicjalizacja modelu
        classifier = CNNImageClassifier(model_name=model_name)
        
        # Pobieranie danych (tylko CIFAR-10 dla szybkości)
        X_train, y_train, X_test, y_test, _ = classifier.download_sample_dataset('cifar10')
        X_train, y_train, X_test, y_test = classifier.preprocess_data(X_train, y_train, X_test, y_test)
        
        # Ograniczenie danych dla szybszego treningu
        X_train_small = X_train[:5000]
        y_train_small = y_train[:5000]
        X_test_small = X_test[:1000]
        y_test_small = y_test[:1000]
        
        # Generatory danych
        train_gen, val_gen = classifier.create_data_generators(X_train_small, y_train_small)
        
        # Budowanie i trening modelu
        classifier.build_model()
        classifier.train_model(train_gen, val_gen, epochs=10)  # Mniej epok dla porównania
        
        # Ocena
        metrics = classifier.evaluate_model(X_test_small, y_test_small)
        results[model_name] = metrics['accuracy']
        
        print(f"Dokładność {model_name}: {metrics['accuracy']:.4f}")
    
    # Wizualizacja porównania
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(models, accuracies, alpha=0.7, color=['blue', 'green', 'red'])
    plt.title('Porównanie dokładności różnych architektur CNN', fontsize=16)
    plt.ylabel('Dokładność')
    plt.ylim(0, 1)
    
    # Dodanie wartości na słupkach
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    classifier, metrics = main()
    
    print("\n" + "="*60)
    response = input("Czy chcesz porównać różne architektury CNN? (y/n): ")
    if response.lower() == 'y':
        comparison_results = compare_models()
        print("\nWyniki porównania:")
        for model, accuracy in comparison_results.items():
            print(f"{model}: {accuracy:.4f}")