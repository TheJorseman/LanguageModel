# LanguageModel
Bengio Language Model From Scratch con un corpus de literatura española de la época dorada. Obras de autores como Miguel de Cervantes Saavedra, Pedro Calderón de la Barca, Lope de Vega, Tirso de Molina.

Código también disponible en [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1A1b6IvK_huclJWCUEhevVEuGEuS0wsqQ?usp=sharing)

## Descripción 

Este es un modelo de lenguaje neuronal basado en el modelo de Bengio (2003). Se propone un mini framework de redes neuronales basado en [Pytorch](https://pytorch.org/) (solo la estructura) y de esta manera tener una estructura flexible para entrenar redes neuronales de forma fácil asi como se muestra en el siguiente código.
```python

model = Sequential(
                Linear(n_input,100, bias=False),
                Linear(100,300), 
                ReLU(),
                BatchNorm1d(),
                Linear(300,n_input),
                Softmax()
                )

loss = CrossEntropyLoss()

optimizer = SGD(model, loss, lr=learning_rate, batch_size=batch_size)
```
También se definen utilidades para entrenar redes neuronales de tipo feed forward. El código esta basado en numpy por lo que la optimización con CUDA NO está disponible y se requiere de poder de cómputo solamente en CPU.

Todo el código se hizo from scratch y se puede consultar la estructura en la carpeta NeuralNetwork.
## Requerimientos

### PIP

Para usar, los requerimientos es necesario tener [pip](https://pip.pypa.io/en/stable/) y ejecutar el siguiente comando.

```bash
pip3 install -r requierements.txt
```

### Anaconda

Con [Anaconda](https://www.anaconda.com/) y ejecutar el siguiente comando.

```bash
conda env create --file languagemodel.yml
```

### Docker
Para probar su funcionamiento en docker se puede ejecutar el archivo bash. Esto creará una imagen y ejecutará un contenedor calculando la probabilidad de una frase por defecto. 

```bash
./build_run.sh
```
O hacer una imagen con el siguiente comando

```bash
docker build -t languagemodel .
```
Eso creará una imagen para después crear tantos contenedores como se requiera.

## Uso 

### Entrenamiento 
```bash
python model.py
```
### Predecir la probabilidad de una frase
```bash
python predict_phrase.py "Frase de prueba"
```

### Modelo pre-entrenado

Para obtener los archivos ya entrenados con 800 epocas estan los archivos model-final.bin, vocabulary, embedding.
Fue entrenado con un corpus reducido debido a que el corpus completo tiene mas de un millon de bigramas para el entrenamiento.
Fueron guardados con pickle por lo que para cargarlos basta con:
```python

vocabulary = pickle.load(open("vocabulary", "rb"))
model = pickle.load(open("model-final.bin", "rb"))
```
Teniendo importadas las clases necesarias para loq ue se quiera hacer.


## Contribuir
Las solicitudes de extracción son bienvenidas. Para cambios importantes, abra un problema primero para discutir qué le gustaría cambiar.

Asegúrese de actualizar las pruebas según corresponda.

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)