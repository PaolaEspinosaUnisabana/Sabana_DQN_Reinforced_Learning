# Maestría en Inteligencia Articial.

# Simulación y Aprendizaje por Refuerzo.

# Grupo:
- valeriagume@unisabana.edu.co
- felipeba@unisabana.edu.co
- angieeshu@unisabana.edu.co
- luisgarccam@unisabana.edu.co
- andresmidi@unisabana.edu.co
- luis.uribe2@unisabana.edu.co
- juanmorme@unisabana.edu.co


# DQN en Acrobot-v1 (Gymnasium)

Agente DQN que resuelve el entorno **Acrobot-v1**.

Documentación: https://gymnasium.farama.org/environments/classic_control/acrobot/

## Entorno
Brazo de dos eslabones colgado de un punto fijo. 
En este problema solo se aplica torque en la articulación central. El objetivo es subir el extremo del segundo eslabón por encima de una línea de referencia.

| Elemento | Descripción |
|---|---|
| **Observaciones** | `Box(6)`: cos θ1, sin θ1, cos θ2, sin θ2, velocidad angular θ1, velocidad angular θ2 |
| **Recompensa** | -1 por cada paso hasta alcanzar la meta (por eso mejor = menos negativo; rango de -500 a ≈ -60) |
| **Terminación** | *terminated*: el extremo libre supera la altura objetivo. *truncated*: se llegan a 500 pasos |


### Acciones

`Discrete(3)`: en cada paso el agente elige un entero que define el torque aplicado en la articulación entre los dos eslabones.

| Acción | Torque | Efecto |
|---|---|---|
| 0 | -1 | Empuja el segundo eslabón en sentido negativo |
| 1 | 0 | No aplica fuerza. El brazo se mueve por inercia y gravedad |
| 2 | +1 | Empuja el segundo eslabón en sentido positivo |

Es importante añadir que solo la articulación central tiene actuador, por lo que el agente no puede levantar el brazo directamente, **debe aprender a balancearlo**, aplicando el torque en el momento justo para ganar energía en cada oscilación. La red tiene 3 salidas, una por acción, y el agente elige la de mayor valor Q.


## El agente
- Red: `6 → 128 → 128 → 3` (ReLU). 
    - Entrada = tamaño de la observación.
    - Salida = un valor Q por acción.

- Replay buffer (50 000), red objetivo (copia cada 500 pasos), epsilon-greedy lineal 1.0 -> 0.05 en 15 000 pasos.
- Cada 10 episodios se evalúa la política greedy (5 episodios) y se guarda el mejor checkpoint.
- Adam (lr 1e-3), γ = 0.99, batch 64, pérdida Huber, 300 episodios, semilla 42.
- Solo `terminated` corta el bootstrap. `truncated` no se trata como estado final.

## Uso
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                 # entrena y guarda modelo, rewards.csv y training_curve.png
python evaluate.py              # evalúa 20 episodios sin exploración
python evaluate.py --render     # ver al agente en ventana
```

## Resultados
![Curva de entrenamiento](training_curve.png)

Evaluación con `evaluate.py` (20 episodios, política greedy, semillas distintas a las usadas para elegir el modelo):

| Métrica | Valor |
|---|---|
| Recompensa media | -77.5 ± 6.7 |
| Mín / Máx | -90 / -63 |

**Nota:** Vemos que DQN es inestable en Acrobot. Durante el entrenamiento la política greedy oscila entre buenos resultados y -500 entre un checkpoint y otro. Por eso `train.py` evalúa cada 10 episodios y **guarda el mejor modelo**, no el último.

## Archivos
- `train.py`: red, entrenamiento y gráfica.
- `evaluate.py`: evaluación del modelo guardado.
- `sabana_dqn_rl.pth`: pesos entrenados.
- `rewards.csv`: recompensa por episodio.
