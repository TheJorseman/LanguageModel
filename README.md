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
También se definen utilidades para entrenar redes neuronales de tipo feed forward. El código esta basado en numpy por lo que la optimización con CUDA no está disponible y se requiere de poder de cómputo solamente en CPU.

Todo el código se hizo from scratch y se puede consultar la estructura en la carpeta NeuralNetwork.
## Uso

Para usar, los requerimientos es necesario tener [pip](https://pip.pypa.io/en/stable/) y ejecutar el siguiente comando.

```bash
pip3 install -r requierements.txt
```

O con [Anaconda](https://www.anaconda.com/) y ejecutar el siguiente comando.

```bash
conda env create --file languagemodel.yml
```

## Uso 

### Entrenamiento 
```bash
python model.py
```
### Predecir la probabilidad de una frase
```bash
python predict_phrase.py "Frase de prueba"
```
## Contribuir
Las solicitudes de extracción son bienvenidas. Para cambios importantes, abra un problema primero para discutir qué le gustaría cambiar.

Asegúrese de actualizar las pruebas según corresponda.

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)