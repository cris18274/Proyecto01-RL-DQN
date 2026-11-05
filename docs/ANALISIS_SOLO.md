# Componente obligatorio: análisis sin implementación

## ¿Por qué BipedalWalker y MuJoCo no se prestan a la Q-table o DQN estándar?

La limitación central no es que dichos entornos sean imposibles de resolver, sino que el **DQN presentado en clase exige un conjunto de acciones discreto**, mientras que tareas de locomoción modernas entregan controles de torque continuos y multidimensionales. Una tabla Q también requeriría discretizar el estado, que suele ser continuo y de alta dimensión.

### Ejemplo de explosión combinatoria

Si un robot tuviera **4** dimensiones de acción continua y cada una se dividiera en **11** niveles, el agente necesitaría **11⁴ = 14 641 acciones discretas** por estado. Con **8** controles a 11 niveles habría **11⁸ = 214 358 881 acciones**. Este es un cálculo *hipotético de discretización*, no una medición de un ambiente Gymnasium en particular; reporta las dimensiones efectivas de la versión de entorno elegida por el usuario si se requiere atribuirlo a BipedalWalker, Hopper, HalfCheetah o Ant.

Para un estado con 17 variables y 10 intervalos por dimensión, una tabla Q requeriría 10¹⁷ estados discretizados *antes* de multiplicar por acciones, claramente impracticable incluso en memoria; aquí **17 es un ejemplo ilustrativo**, no un dato afirmado de ningún entorno específico.

### Métodos más adecuados

- **PPO:** optimiza una política parametrizada con regularización de cambios en política. Puede modelar distribuciones continuas de acciones; suele ser una opción robusta con múltiples entornos paralelos.
- **SAC:** aprendizaje off-policy actor-critic con regularización de entropía; combina un crítico con una política estocástica continua, favoreciendo reutilización de experiencias.
- **TD3:** actor-critic determinista para acciones continuas, con críticos gemelos y otros mecanismos para reducir sobreestimación.

Estas son familias pertinentes porque permiten **acciones continuas sin enumerar todas las combinaciones posibles**. Elegir entre ellas también depende de estabilidad, costo de simulación, exploración y paralelismo.

### ¿Por qué el tabular falla con observaciones tipo Atari?

Los píxeles tienen dimensiones muy altas, con muchos valores de intensidad. Tratar cada imagen posible como un estado separado conduce a una tabla astronómica y casi nula generalización entre imágenes parecidas. Una CNN compartida aprende representaciones espaciales y reutiliza parámetros; eso es una ventaja de aproximación funcional, aunque la naturaleza no estacionaria del entrenamiento DQN introduce nuevas dificultades.

## Dos detalles de rigor

1. El número exacto de dimensiones cambia con la **versión del entorno** (y con wrappers). Confírmalo en la documentación y en `env.action_space` y `env.observation_space` antes de atribuir cifras a nombres como `Ant-v5`.
2. Ninguno de los argumentos obliga a afirmar que “PPO siempre supera a SAC” o que “DQN nunca funciona con acciones discretizadas”: se trata de tradeoffs de escalabilidad, no teoremas universales.
