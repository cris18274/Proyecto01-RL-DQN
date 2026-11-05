# Defensa oral: modificaciones en vivo (guía de seguridad)

**Antes de tocar código:** muestra la función original, verbaliza hipótesis y solo luego edita. Mantén un commit limpio. Las demostraciones de 320–2000 pasos no sirven como evidencia científica. No uses `--overwrite` sobre `results/tier1`, `results/tier1_200k` ni `results/tier2`.

## Mapa de código (sin necesidad de buscador)

- `rl01/cli.py:36-55`: semillas, alpha, gamma, epsilon para cada entrenamiento.
- `rl01/cli.py:58-77`: Tier1, tres discretizaciones y DQN.
- `rl01/cli.py:80-102`: Tier2, LR, target cadences, ablación y Double DQN opcional.
- `rl01/tabular.py:12-17`: rangos de discretización; `:20-39`: codificar estados; `:62-68`: actualización Q.
- `rl01/dqn.py:13-25`: hiperparámetros; `:28-38`: red 128-128; `:41-72`: replay; `:91-125`: epsilon-greedy, Q(s,a), target, Huber y backprop.
- `rl01/training.py:22-36`: evaluación greedy; `:99-120`: logs; `:123-166`: entrenamiento, truncado, sincronización y holdout final.
- `rl01/analysis.py:38-76`: tabla por semillas; `:79-129`: curvas ±1 SD.

## Cambios concretos que un profesor podría pedir

| Solicitud | Cambio seguro | Predicción/defensa |
|---|---|---|
| Épsilon final 0.10 → 0.05 | En `rl01/cli.py`, dentro de `tier2`, `eps_end=0.10` → `eps_end=0.05`. | Explotación final mayor, menos exploración tardía; no garantizar mejora. |
| Decay 80% → 50% | En la misma llamada `eps_decay=0.80` → `0.50`. | Llega antes al epsilon mínimo; más episodios finales explotadores. |
| Añadir 12 bins | En `tier1`, `(6, 10, 20)` → `(6, 10, 12, 20)`, crear salida nueva. | 12^4 = 20,736 estados; peor cobertura si pasos insuficientes. |
| Limitar velocidad angular | Cambiar valores en `CARTPOLE_CLIP_RANGES`, o `--include-range-study`. | Mayor resolución local, aliasing/saturación de estados extremos. |
| LR 0.0005 → 0.0001 | Agregar una variante `replace(base, learning_rate=1e-4)` con nombre nuevo. | Gradientes más conservadores y posible convergencia más lenta. |
| Target 1,000 → 500 | Agregar variante `('target_500', replace(base, target_update_frequency=500))`. | Valores objetivo más frescos, pero mayor variación del objetivo. |
| Sin target network | Ya disponible: `('ablation_no_target', replace(base, disable_target=True))`. | `bootstrap_net=self.online`, con `no_grad`, objetivo no congelado entre actualizaciones. |
| Sin replay | **No** se logra poniendo `buffer_capacity=1` con batch 64. Requiere cambiar pipeline de actualizaciones para entrenar transiciones recientes y mantener diseño comparable. | No simular una ablación inválida. |
| Batch de 64 a 32 | Agregar `replace(base, batch_size=32)` en un estudio aparte. | Gradientes más ruidosos y más actualizaciones viables temprano. |
| Red 128 → 64 | Agregar `replace(base, hidden_width=64)`; el output sigue igual. | Menos capacidad y menor costo; no implica mejor o peor automáticamente. |
| Huber por MSE | `rl01/dqn.py`: `F.smooth_l1_loss` → `F.mse_loss`, luego restaurar. | Más sensibilidad a TD errors grandes. |
| Muestreo sin reemplazo | `np.random.Generator.choice(self.size,size=batch_size,replace=False)` en replay (ya hay verificación `self.size>=batch_size`). | Evita repetidos dentro de un minibatch; cambia distribución de muestreo. |
| Descuento 0.99 → 0.90 | `gamma` en `DQNConfig` y/o llamada a tabular. | Da menos peso a recompensa distante; puede empeorar Acrobot. |
| Número de semillas | `--seeds 11,22,33,44` y salida nueva. | Mejor estimación variabilidad, más cómputo. |
| Evaluar 20 en vez de 10 episodios | CLI `--eval-episodes 20`, sin editar código. | Checkpoint más preciso pero mayor tiempo de evaluación. |
| Double DQN | CLI `--include-double` (en carpeta aparte) o `double_dqn=True`; líneas 112-117 DQN. | Online selecciona `argmax`, target valora; mitiga sobreestimación. |

## Demostración segura para mañana

Desde la raíz del proyecto, sin modificar datos del paper:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m rl01.cli smoke --out "results/live_smoke_$(Get-Date -Format yyyyMMdd_HHmmss)"
.\.venv\Scripts\python.exe -m rl01.cli analyze --tier-dir results/tier2
```

El último comando regenera las curvas originales desde CSV. Para demostrar la reproducibilidad de todas las figuras del paper SIN tocarlas, ejecuta también `python report/make_paper_figures.py` (guarda archivos nuevos en `report/figures_reproduced`).

Para inspeccionar sin ejecutar entrenamiento largo:

```powershell
.\.venv\Scripts\python.exe -c "from rl01.tabular import Discretizer; d=Discretizer(10); print('Estados:',d.n_states, 'Estado s0:',d.encode([0,0,0,0]))"
Get-Content .\results\tier2\summary_by_variant.csv | Select-Object -First 8
```

## Cinco trampas frecuentes

1. Retorno (`episode return`) no es accuracy ni F1. En Acrobot, -88.39 es mejor que -500.
2. Un checkpoint de 10 episodios no demuestra el éxito final de 100 episodios; tampoco una única semilla.
3. `terminated=True` anula el bootstrap; `truncated=True` termina el episodio, pero no anula ese término en el cálculo implementado.
4. Un target update cada 250 se mide en **pasos del entorno**, no en pasos de gradiente; DQN optimiza cada 4 pasos tras `learning_starts=1000`.
5. `git init` no publica en GitHub. Solo `git push` al remoto crea la copia visible; verifica URL en navegador.
