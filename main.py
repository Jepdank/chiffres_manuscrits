import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normaliser les données
x_train = x_train / 255.0
x_test = x_test / 255.0

# Construction du modèle
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrainement du modèle
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nPrécision du modèle:', test_acc)

# Prédictions sur le images de test
predictions = model.predict(x_test)

# Affichage des erreurs
def plot_wrong_predictions(x_test, y_test, predictions):
    wrong = []
    for i, (true, pred) in enumerate(zip(y_test, predictions.argmax(axis=1))):
        if true != pred:
            wrong.append((i, true, pred))
    print(f"Total d'erreurs: {len(wrong)}")

# Afficher les 9 premières erreurs
    plt.figure(figsize=(10, 10))
    for i, (index, true, pred) in enumerate(wrong[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[index], cmap='gray')
        plt.title(f"Vrai: {true} / Prédit: {pred}")
        plt.axis('off')
    plt.show()

plot_wrong_predictions(x_test, y_test, predictions)