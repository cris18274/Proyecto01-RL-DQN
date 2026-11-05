# Esqueleto para artículo de 6–8 páginas, dos columnas

**Título sugerido:** *Sample-Efficiency, Discretization and Stability in Tabular Q-Learning and Deep Q-Networks for Gymnasium Control Tasks*

**Abstract (rellenar al disponer de datos):** Problema de aproximación tabular frente a función Q neuronal; entornos; variantes; 3+ semillas; misma cantidad de pasos; resultados de retorno final y estabilidad; límites. **No insertar números que no provengan de `results/`.**

## 1. Introducción (0.5–0.75 pág.)

Expón dilema memoria/localización vs generalización; justificación de comparación controlada en CartPole y entornos de control con acciones discretas más complejos. Formula preguntas:

- RQ1: ¿Cómo influye la granularidad de discretización en eficiencia y estabilidad del tabular?
- RQ2: ¿En qué condiciones mejora DQN frente a una tabla con los mismos pasos del entorno?
- RQ3: ¿Qué impacto tienen learning rate y red objetivo sobre los resultados en Acrobot?

## 2. Pertinencia de algoritmos (0.5 pág.)

Estado y acción por entorno; razón por la que DQN estándar no maneja torques multidimensionales continuos; resumen cuantificado de combinatoria (ver `ANALISIS_SOLO.md`).

## 3. Metodología (1.0–1.5 págs.)

Describe MDP, Bellman y target Q, redes 128×128, epsilon greedy, replay 50k, batch 64, sync target, Huber y Adam. Fija versiones reales de `config.json`, CPU, seeds, límites de episodios, intervalos de evaluación, rangos de bins, justificación de clipping y manejo de `terminated` frente a `truncated`.

## 4. Protocolo experimental (0.75–1 pág.)

- Tier 1: CartPole; 3 granulaciones; DQN; 80k o los pasos realmente utilizados, idénticos.
- Tier 2: Acrobot o LunarLander; LR 0.00025 / 0.0005 / 0.001, target sync 250 / 1000 / 4000; ablación sin red objetivo.
- Semillas: reporta concretamente las que ejecutaste. Evalua 10 episodios greedy cada N pasos; 100 episodios *final_holdout*.
- Métricas: retorno medio, dispersión entre semillas, costo de CPU y primer cruce de umbral **exploratorio**.

## 5. Resultados (1–1.5 págs.)

Inserta figuras `results/tier1/learning_curves.pdf` y `results/tier2/learning_curves.pdf`, luego tabla `summary_by_variant.csv`. Anota si el umbral de la consigna se cumplió *según la evaluación final*, no solo en una instantánea intermedia. Muestra tiempos de CPU y variabilidad.

## 6. Discusión (0.75–1 pág.)

Relaciona curvas con Bellman, aproximación funcional, sesgo de bins, replay/target; identifica problemas de convergencia; evita confundir correlación con mecanismo demostrado; discute tres semillas y OFAT como limitaciones.

## 7. Análisis sin implementación (1–2 págs. si la consigna lo exige como sección independiente)

BipedalWalker, locomoción MuJoCo, acciones continuas, explosión combinatoria, PPO/SAC/TD3, Atari y generalización. Integra esta sección ajustando el presupuesto de páginas total.

## 8. Conclusión, declaración IA y reproducibilidad

Las conclusiones solo pueden redactarse después de ejecutar todas las variantes. Adjunta enlace al repositorio, comandos reproducibles, declaración IA y listado de prompts principales, revisado para reflejar el trabajo real.
