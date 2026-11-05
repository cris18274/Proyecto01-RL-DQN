# Defensa técnica oral: preguntas que debes saber responder

**Contexto:** la consigna asigna la mitad de la nota a una defensa de 10 minutos y una caminata de 10 minutos por el código. Esto es una guía de estudio y ejemplos, no resultados de entrenamiento.

## A. Recorre la arquitectura en este orden

1. `rl01/cli.py`: construye los experimentos, fija semillas y mantiene el mismo número de pasos por variante.
2. `rl01/tabular.py`: discrétiza estado continuo, crea tabla Q, selecciona acción y aplica la regla de Bellman.
3. `rl01/dqn.py`: red neuronal que predice un valor Q por acción; memoria de experiencias; red objetivo; actualización de gradientes.
4. `rl01/training.py`: control del episodio, epsilon, logs, evaluación cada 10 000 pasos y holdout final.
5. `rl01/analysis.py`: medias y bandas entre semillas, curvas individuales y tiempos.
6. `rl01/video.py`: reconstruye un agente con sus pesos y renderiza una trayectoria real.

## B. Preguntas y respuestas difíciles

**1. ¿Por qué Q-learning tabular sirve para CartPole si el estado es continuo?**

No se aplica directamente sobre el vector continuo; definimos una función explícita de discretización que transforma el vector de cuatro dimensiones en un índice finito. Así aprendemos una estimación Q por estado discretizado y acción. La aproximación pierde resolución y puede fusionar estados de dinámicas diferentes.

**2. ¿Cuántos valores Q existen con 20 bins por dimensión en CartPole?**

Hay 20⁴ = 160 000 estados discretizados y dos acciones; por tanto, 320 000 valores Q en la tabla. El crecimiento es exponencial respecto del número de variables discretizadas.

**3. ¿Por qué no usamos una regresión lineal convencional?**

Porque la tarea es tomar decisiones secuenciales y optimizar un retorno futuro acumulado. La red ajusta Q(s,a) usando objetivos temporales de Bellman, no una etiqueta fija i.i.d. de aprendizaje supervisado. Una aproximación lineal de Q sería posible, pero tendría distintas limitaciones de representación.

**4. ¿Qué predice exactamente la red DQN?**

Un vector de tamaño igual al número de acciones discretas. Cada componente aproxima el retorno futuro descontado de seleccionar una acción en el estado observado y después actuar según la política implícita en Q.

**5. ¿Por qué la última capa NO tiene softmax?**

Los valores Q no son probabilidades; pueden ser negativos y no necesitan sumar uno. Se elige `argmax`, pero la política de exploración intercala acciones aleatorias.

**6. ¿Qué significa gamma = 0.99?**

Que una recompensa recibida un paso después pesa 0.99 respecto a una recompensa inmediata. Es un factor de descuento; gamma cercano a 1 favorece consecuencias a largo plazo. No indica 99 % de precisión.

**7. ¿Qué hace alpha en Q-learning?**

Controla cuánto de la diferencia entre objetivo temporal y valor actual incorporamos en la tabla. Si es demasiado alto puede generar fluctuaciones; demasiado bajo puede retardar el aprendizaje.

**8. ¿Qué diferencia hay entre alpha tabular y learning rate de Adam?**

Alpha actualiza directamente una entrada Q mediante TD error. El learning rate de Adam afecta la magnitud de los pasos de optimización de todos los pesos de la red mediante gradientes adaptativos.

**9. ¿Por qué utilizamos Huber loss?**

Es cuadrática para errores pequeños y menos agresiva con errores extremos que MSE. Reduce sensibilidad de las actualizaciones cuando hay TD errors grandes; no garantiza estabilidad global.

**10. ¿Qué problema resuelve replay buffer?**

La transición de estados secuenciales está temporalmente correlacionada. El buffer mezcla experiencias de momentos distintos al muestrear mini-lotes, disminuyendo esa dependencia y reutilizando observaciones. No convierte las muestras en perfectamente independientes.

**11. ¿Por qué hay dos redes?**

Una online se actualiza por gradiente; otra target produce objetivos que permanecen relativamente estables durante un intervalo. Sin ello, cambian simultáneamente la predicción y el objetivo que perseguimos.

**12. ¿Qué ocurre exactamente en `disable_target=True`?**

`bootstrap_net` pasa a ser la misma red online que se está entrenando. Conservo replay, batch, learning rate y arquitectura: cambio una variable causal relevante. Si el rendimiento empeora, la interpretación es compatible con mayor inestabilidad del bootstrap, pero no demuestra por sí sola causalidad universal.

**13. ¿Por qué `with torch.no_grad()`?**

Porque los objetivos de Bellman son referencias fijas para *esta* actualización de gradiente. No queremos que el error retropropague a través de la construcción del objetivo. Sí propagamos a través de la predicción Q(s,a) de la red online.

**14. ¿Qué hace `gather(1, actions.unsqueeze(1))`?**

Selecciona, para cada observación del batch, el Q correspondiente a la acción realmente ejecutada. El `max` de la siguiente observación se usa solo para construir el objetivo (vanilla DQN).

**15. ¿Qué diferencia hay entre `terminated` y `truncated`?**

`terminated` indica llegada a un terminal definido por la dinámica del problema (incluido fracaso/éxito). `truncated` indica interrupción externa, como llegar a un límite de pasos. En este código el bootstrap se anula solo ante `terminated`; ambos obligan a reiniciar el episodio.

**16. ¿Qué significa off-policy?**

La política de comportamiento explora con ε-greedy, mientras el objetivo de Q-learning usa el valor máximo de la política greedy. No se presupone que las acciones generadoras sean exactamente las acciones óptimas.

**17. ¿Por qué la salida de la red es 2 en CartPole y 3 en Acrobot?**

Porque los espacios de acción respectivos son discretos con esos tamaños. La entrada, en cambio, tiene 4 observaciones CartPole y 6 Acrobot. El código infiere ambas dimensiones del entorno.

**18. ¿Qué es un paso de entorno?**

Una llamada a `env.step(action)`. No equivale a un episodio ni a una actualización de pesos; la red se optimiza cada cuatro pasos después del calentamiento.

**19. ¿Por qué no comparar solo episodios?**

Los episodios pueden tener longitudes muy diferentes; el número de interacciones con el entorno es un proxy más homogéneo de eficiencia muestral. El tiempo real sigue importando para eficiencia computacional.

**20. ¿Qué valida 100 episodios finales?**

Una estimación empírica del rendimiento promedio bajo la política greedy en estados iniciales de prueba, separada de los checkpoints. No es una garantía de rendimiento en todos los estados ni un intervalo estadístico que se estreche mágicamente al multiplicar episodios correlacionados.

**21. ¿Por qué usar tres semillas y no una?**

Inicialización, exploración, muestreo de replay y trayectorias pueden variar entre corridas. Tres semillas es el mínimo de la consigna; cinco serían preferibles para un panorama más confiable. Un éxito aislado no representa robustez.

**22. ¿Qué indican las bandas en los gráficos?**

Muestran ±1 desviación estándar de los **retornos medios por semilla** en cada checkpoint, no un intervalo de confianza sobre todos los pasos de todos los episodios mezclados.

**23. ¿Con esos tres seeds puedo asegurar superioridad estadística?**

No. Las comparaciones son exploratorias; un intervalo puede ser amplio y la potencia estadística pequeña. Presentaría medias, dispersión, intervalos y diferencias pareadas, sin afirmar causalidad fuerte ni victoria universal.

**24. ¿Por qué variar un parámetro por vez?**

Porque es el diseño solicitado: permite describir sensibilidad local de LR y frecuencia de red objetivo respecto de un baseline documentado. No identifica interacciones y se debe reconocer esa limitación.

**25. ¿Por qué Double DQN reduce sobreestimación?**

En DQN vanilla el máximo tanto selecciona como evalúa una acción, favoreciendo acciones con estimaciones accidentalmente altas. Double DQN separa selección (`online`) de estimación (`target`). El efecto es contextual, no garantía de mejor puntuación.

**26. ¿Qué significa que DQN no haya llegado a −100 en Acrobot?**

Que bajo ese presupuesto, algoritmo y semillas el criterio del docente no fue alcanzado. Diagnosticaría la curva, la exploración, la varianza, la pérdida, la actualización objetivo, el efecto de ablation y los límites de tiempo antes de atribuirlo a una única causa.

**27. ¿Dónde se ven los pasos totales y el reloj?**

En `summary.json`: `train_steps`, `total_evaluation_steps_excluded_from_train_budget` y `wall_clock_seconds_including_evaluation`. Se evita confundir interacciones de entrenamiento con cómputo total.

**28. ¿Qué tendría que hacer para reproducir una figura?**

Instalar dependencias, ejecutar la configuración con las semillas indicadas y usar `python -m rl01.cli analyze --tier-dir results/tier1` o `results/tier2`; el script grafica los CSV reales, no números insertados manualmente.

**29. ¿Qué es la *deadly triad*?**

Bootstrapping + aprendizaje off-policy + aproximación funcional. En combinación pueden generar divergencia o inestabilidad bajo ciertas condiciones, y explican por qué los mecanismos estabilizadores de DQN son tan importantes.

**30. ¿Qué puede concluirse si el tabular supera a DQN en CartPole?**

Que bajo un presupuesto fijo y discretización apropiada una tabla puede aprender más rápido en un entorno de baja dimensión. No concluiría que lo tabular vence a deep RL en general; DQN puede aportar generalización y escala a observaciones continuas más ricas.

## C. Cambios en vivo que podrían pedirte

| Solicitud del profesor | Dónde editar | Qué predices |
|---|---|---|
| Bajar ε final de 0.10 a 0.01 | `rl01/cli.py`: `eps_end` de Tier 2 | Más explotación al final; podría reducir descubrimiento tardío |
| Aumentar bins 10 a 20 | `rl01/cli.py`: `for bins in ...` | Más resolución y muchas más entradas Q; mayor costo de exploración |
| Cambiar alpha | `train_one(alpha=...)` desde `cli.py` | Tabla más reactiva o más conservadora frente a TD errors |
| Sin red objetivo | `replace(base, disable_target=True)` | Objetivos móviles, posible mayor variabilidad |
| Activar Double DQN | `--include-double` y `dqn.py` | Reduce sesgo de maximización, quizá a costa de convergencia distinta |
| Cambiar target 1000 a 250 | `replace(base, target_update_frequency=250)` | Objetivos más frescos, pero menos estacionarios |
| Frecuencia de optimización 4 a 1 | `DQNConfig(train_frequency=1)` | Más gradientes por muestra, mayor costo temporal y riesgo de sobreajuste de replay |
| Eliminar truncación del criterio de reinicio | `rl01/training.py` | **No hacerlo:** sería un bug; `truncated` sí finaliza el episodio |
| Mostrar acciones concretas de DQN | `DQNAgent.act()` | `online(state)` retorna el vector Q, luego `argmax` |

## D. Frases para interpretar resultados sin inventar causalidad

- *“La tendencia es consistente con mayor inestabilidad al eliminar la target network, pero con N semillas reducido se interpreta como evidencia exploratoria.”*
- *“No medimos accuracy ni F1: medimos retornos acumulados por episodio, eficiencia muestral y variabilidad entre semillas.”*
- *“La banda no es incertidumbre de los pesos: refleja dispersión de las corridas independientes.”*
- *“El resultado de 10 episodios por checkpoint monitoriza aprendizaje; la decisión sobre cumplimiento se toma con 100 episodios de evaluación final.”*
- *“La elección de rangos tabulares implica un sesgo inductivo; explicamos saturación, resolución y tradeoff estadístico.”*
