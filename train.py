import tensorflow as tf
from tensorflow import keras # type: ignore

# Dados MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# ↑ Carrega MNIST: 
#   X_train = 60.000 imagens de treino
#   y_train = labels (0-9) correspondentes
#   X_test = 10.000 imagens de teste
#   y_test = labels de teste
X_train = X_train / 255.0
X_test = X_test / 255.0
# ↑ Normaliza: divide por 255 para valores ficarem entre 0 e 1
#   (em vez de 0-255)

# Modelo
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # ↑ Transforma imagem 28x28 num vector de 784 valores
    
    keras.layers.Dense(128, activation='relu'),
    # ↑ Camada com 128 neurónios, activation relu
    
    keras.layers.Dense(10, activation='softmax')
    # ↑ Camada final com 10 neurónios (0-9 dígitos)
    #   softmax = converte em probabilidades
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# ↑ Configura treino:
#   optimizer = algoritmo de aprendizagem (adam = bom)
#   loss = função de erro (sparse_categorical = para classificação)
#   metrics = mostra accuracy durante treino

model.fit(X_train, y_train, epochs=10, batch_size=32)
# ↑ Treina o modelo:
#   X_train, y_train = dados
#   epochs=10 = passa 10 vezes pelos dados
#   batch_size=32 = processa 32 imagens de cada vez

model.save('models/modelo_digitos.keras')
# ↑ Guarda modelo em formato .keras
print("Modelo guardado em models/modelo_digitos.keras")
