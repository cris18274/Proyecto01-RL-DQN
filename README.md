# Proyecto 01 — Q-learning tabular vs DQN en Gymnasium

**Código experimental desde cero (Python + PyTorch) / Defensa de Agentes Inteligentes**

> **ACTUALIZACIÓN DE ENTREGA (05/11/2026):** este repositorio ahora SÍ incluye los resultados reales auditados de **42 entrenamientos** (Tier1 80k, Tier1 200k, Tier2), **5.16 millones de pasos de entrenamiento**, CSV, pesos, figuras y **cuatro GIF**. Incluye también `report/Paper_RL_DQN_Revisado.pdf` y `slides/Defensa_RL_DQN_Resultados_Reales.pptx`. La sección siguiente documenta el protocolo reproducible de entrenamiento y evaluación. Ver `RESULTADOS_VERIFICADOS.md`, `docs/AUDITORIA_RUBRICA.md` y `docs/GITHUB_PUBLICACION.md` para el estado final.

**Publicación:** `git init` y `git push` todavía deben ejecutarse desde la cuenta del estudiante. Enlace remoto aún por completar. **Campos pendientes en el paper:** nombre y correo del autor, URL real del repositorio y validación de prompts registrados. No uses el ZIP de 249 MB con `.venv` para publicar.

El proyecto implementa exactamente el núcleo obligatorio de la consigna recibida:

- **Tier 1:** `CartPole-v1`, Q-learning tabular con **6, 10 y 20 bins por dimensión**, comparado con DQN mediante el mismo presupuesto de pasos de entrenamiento. Estudio adicional opcional de rangos de discretización.
- **Tier 2:** `Acrobot-v1`, DQN; barridos **one-factor-at-a-time (OFAT)** de learning rate y frecuencia de actualización de la red objetivo (3 valores por parámetro, incluyendo el valor base); ablación **sin red objetivo**.
- **Tier 3 opcional (obligatorio si son dos estudiantes):** Double DQN sobre el mismo Tier 2.
- **Calidad experimental:** mínimo tres semillas; evaluación greedy sin exploración cada 10 000 pasos; curvas con media ± desviación estándar entre semillas; evaluación final independiente sobre 100 episodios; registro de tiempo, pasos, configuración, versiones del entorno, CSV, pesos, GIF real del agente.

**Importante (actualizado):** además de la implementación, este paquete entrega resultados reales ya ejecutados. El criterio final se alcanzó en Acrobot para `target_250` (3/3 semillas), pero no en la media de CartPole. Las pruebas internas sintéticas solo comprueban integración y **no constituyen resultados científicos**.

## 0. Evidencia incluida y política de reproducción

Los entrenamientos ya fueron ejecutados y sus resultados están incluidos en `results/tier1`, `results/tier1_200k` y `results/tier2`. **No vuelvas a ejecutar** los scripts `03_tier1.ps1` o `04_tier2.ps1` sobre las mismas rutas sin un plan: el CLI detectará carpetas existentes y emitirá `FileExistsError` para no sobrescribir evidencia. Para reproducir entrenamientos en carpetas nuevas usa `--out results/reproduccion_tier1_80k` o `--out results/reproduccion_tier2_100k`. Para regenerar las figuras sin volver a entrenar, usa los comandos `analyze` de la Sección 4.

El manuscrito revisado está en `report/` y las 24 diapositivas en `slides/`. Revisa `docs/AUDITORIA_RUBRICA.md` para el alcance de las afirmaciones y `docs/PENDIENTES_EDITORIALES.md` antes de entregar.

## 1. Iniciar en Windows PowerShell

Descomprime el ZIP. Abre PowerShell en la carpeta principal `Proyecto01_RL_DQN` (donde está este archivo). Si no tienes `py`, sustituye `py -3` por `python`.

```powershell
# Primero crea .venv, instala dependencias y ejecuta las pruebas unitarias.
powershell -ExecutionPolicy Bypass -File windows/01_instalar.ps1

# Prueba de ~320 pasos por variante: verifica ejecución real en Gymnasium.
# Los resultados de esta prueba NO se utilizan en el paper.
powershell -ExecutionPolicy Bypass -File windows/02_prueba_rapida.ps1
```

Alternativa manual:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m rl01.cli smoke --overwrite
```

**Si tu PC requiere una política de ejecución especial, no necesitas activar el entorno:** todos los comandos invocan directamente el `python.exe` de `.venv`.

## 2. Tier 1: CartPole (obligatorio)

```powershell
.\.venv\Scripts\python.exe -m rl01.cli tier1 --seeds 11,22,33 --steps 80000 --eval-every 10000
```

Variantes con exactamente 80 000 pasos de entrenamiento por semilla:

| Nombre de variante | Algoritmo | Hiperparámetro controlado |
|---|---|---|
| `tab_bins6` | Q-learning | 6 bins por dimensión; 6⁴ = 1 296 estados discretos |
| `tab_bins10` | Q-learning | 10 bins; 10⁴ = 10 000 estados |
| `tab_bins20` | Q-learning | 20 bins; 20⁴ = 160 000 estados |
| `dqn_base` | DQN | Red de dos capas ReLU de 128 unidades; replay y target network |

Para estudiar también **los rangos** en forma controlada:

```powershell
.\.venv\Scripts\python.exe -m rl01.cli tier1 --seeds 11,22,33 --steps 80000 --eval-every 10000 --include-range-study --overwrite
```

Esta bandera agrega `tab_bins10_tight`, con rangos de velocidades más estrechos (±1.5 y ±2.0 frente a ±3.0 y ±3.5). **Advertencia:** ejecutará nuevamente todas las variantes con `--overwrite`; guarda primero los resultados anteriores si no quieres reemplazarlos.

El material del profesor menciona **200 000 pasos** como ejemplo de presupuesto en CartPole. Si hay disponibilidad computacional, aumenta `--steps 200000` manteniendo **igual presupuesto para todas las variantes** y documenta el cambio antes de analizar resultados. Los 80 000 pasos de este paquete son un punto de partida de ejecución, no una garantía de convergencia.

## 3. Tier 2: Acrobot (obligatorio)

```powershell
.\.venv\Scripts\python.exe -m rl01.cli tier2 --environment Acrobot-v1 --seeds 11,22,33 --steps 100000 --eval-every 10000
```

| Configuración | Qué varía respecto a base |
|---|---|
| `dqn_base` | LR 0.0005, target update 1 000 pasos de entorno |
| `lr_0p00025` | LR 0.00025; todo lo demás idéntico |
| `lr_0p001` | LR 0.001; todo lo demás idéntico |
| `target_250` | Sincronización objetivo cada 250 pasos |
| `target_4000` | Sincronización objetivo cada 4 000 pasos |
| `ablation_no_target` | Desactiva target network; mantiene arquitectura, replay, LR y entrenamiento |

De manera opcional, usa LunarLander como escenario Tier 2 (requiere Box2D):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-box2d.txt
.\.venv\Scripts\python.exe -m rl01.cli tier2 --environment LunarLander-v3 --steps 100000 --out results/tier2_lunar
```

### Tier 3 (solo si corresponde)

```powershell
.\.venv\Scripts\python.exe -m rl01.cli tier2 --environment Acrobot-v1 --seeds 11,22,33 --steps 100000 --include-double --out results/tier2_double
```

El flag **agrega** `tier3_double_dqn` (selección de acción usando red online y estimación de valor usando red objetivo), en las mismas condiciones de Tier 2. No añade otras extensiones de Tier 3.

## 4. Archivos resultantes

Después de completar Tier 1 se obtiene, por ejemplo:

```text
results/tier1/
  tab_bins6_seed0011/
    config.json              # hiperparámetros + versiones de librerías y hardware
    train_episodes.csv       # retornos de entrenamiento y epsilon
    evaluations.csv          # retorno greedy por checkpoint, separadamente de training
    eval_episodes.csv        # cada retorno de evaluación para auditoría
    policy.npz               # tabla Q entrenada
    summary.json             # resultado final, tiempo y umbral
  dqn_base_seed0011/
    policy.pt                # pesos PyTorch (state_dict)
    ...
  final_by_seed.csv
  summary_by_variant.csv
  summary_by_variant.json
  paired_vs_dqn_base.csv    # contraste pareado por semilla (exploratorio)
  learning_curves.png        # media +/- 1 SD entre semillas
  learning_curves.pdf
  individual_seeds.png
```

Los checkpoints de **10 episodios** se usan para las **curvas**; el resultado final para comprobar el éxito proviene de **100 episodios holdout** que usan semillas distintas. Los pasos de evaluación quedan fuera del presupuesto de entrenamiento, pero **sí** se contabilizan dentro del tiempo de reloj. El primer cruce de umbral en checkpoints, cuando existe, es **exploratorio**, no certifica éxito en 100 episodios.

Si ya tienes resultados y deseas regenerar figuras:

```powershell
.\.venv\Scripts\python.exe -m rl01.cli analyze --tier-dir results/tier1
.\.venv\Scripts\python.exe -m rl01.cli analyze --tier-dir results/tier2
```

Cada figura se reproduce desde los CSV reales guardados; las curvas y retornos del estudio original **sí** están incluidos en este repositorio.

**Figuras del paper (reproducir todas con un solo comando, sin entrenar):**

```powershell
.\.venv\Scripts\python.exe .\report\make_paper_figures.py
```

Este script escribe en `report/figures_reproduced/` (sin sobrescribir las imágenes incluidas) y genera `tier1_curves_light.png`, `paper_cartpole_budget.png`, `tier2_curves_light.png` y `tier2_bars_light.png` (más una curva de CartPole 200k) directamente desde `results/*/evaluations.csv` y `results/*/summary_by_variant.csv`; usa exactamente los mismos datos, aunque el diseño visual puede cambiar de la figura original. Para regenerar **el PDF** del artículo se requiere una distribución LaTeX: desde `report/`, `latexmk -pdf paper_rl_dqn_revisado.tex`.


## 5. Grabar videos/GIF del agente realmente entrenado

```powershell
.\.venv\Scripts\python.exe -m rl01.cli video --run-dir results/tier1/dqn_base_seed0011 --out results/tier1/cartpole_dqn.gif
.\.venv\Scripts\python.exe -m rl01.cli video --run-dir results/tier2/dqn_base_seed0011 --out results/tier2/acrobot_dqn.gif
```

Se necesita renderizador `rgb_array` y la dependencia `pygame` instalada con `gymnasium[classic-control]`. Los GIF se generan a partir de episodios reales del agente guardado; NO son animaciones ilustrativas.

## 6. ¿Cuánto entrenamiento corre realmente?

Con los valores por defecto y tres semillas independientes:

- Tier 1: 4 variantes × 3 semillas × 80 000 = **960 000 pasos de entrenamiento**.
- Tier 2: 6 variantes × 3 semillas × 100 000 = **1 800 000 pasos de entrenamiento**.
- Total: **2 760 000 pasos** de entrenamiento, más evaluaciones periódicas y 100 episodios finales por corrida. La duración real depende de hardware y longitud de episodios: **mide el tiempo, no prometas una duración**.

Para verificar rápido una máquina limitada, puedes reducir `--steps` en una corrida exploratoria en otra carpeta. No mezcles corridas con presupuestos distintos dentro del mismo gráfico comparativo. Si una configuración no alcanza el umbral, explica qué muestran los datos: la consigna admite resultados negativos fundamentados.

## 7. Parámetros y fórmulas que tienes que defender

**Tabular:** `rl01/tabular.py`. Se aplica Q-learning off-policy:

\[Q(s,a)\leftarrow Q(s,a)+\alpha\left(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right).\]

Los rangos de velocidad de CartPole no están finitamente acotados en la definición oficial; aquí se recortan a intervalos de ingeniería documentados. Esto introduce aliasing/saturación en extremos y debe discutirse como limitación.

**DQN:** `rl01/dqn.py`. Red neuronal `(obs_dim, 128, 128, n_actions)` con ReLU, pérdida Huber, replay buffer, selección ε-greedy, red objetivo, clipping de norma de gradiente. Objetivo:

\[y=r+\gamma(1-\texttt{terminated})\max_{a'}Q_{\theta^-}(s',a').\]

El final de episodio `truncated=True` por límite de tiempo **no equivale** a un terminal verdadero para el bootstrap; la iteración finaliza, pero el objetivo temporal puede conservar el valor futuro. Selección de ε-greedy aleatoria no es lo mismo que evaluación greedy ε=0.

**Double DQN opcional:** `argmax` en `online`, valoración en `target`, para reducir sobreestimación derivada del operador máximo. No equivale a usar dos redes independientes.

## 8. Reproducibilidad, limitaciones y política IA

- Se almacenan `seed`, versiones de Python/Gymnasium/PyTorch, hardware, hiperparámetros y logs por corrida.
- Se utilizan semillas de entorno, Python, NumPy y PyTorch; esta práctica mejora reproducibilidad, **pero no garantiza resultados bit a bit entre equipos**.
- Comparar las **medias por semilla** no equivale a independencia por episodio: se muestran además las trayectorias individuales. Con tres semillas la estimación de incertidumbre es frágil.
- Se aplica OFAT: evalúa sensibilidad **local** y no estima interacciones entre hiperparámetros.
- Es una implementación para aprendizaje y defensa oral, no una afirmación de estado del arte. No se aplica ajuste extensivo de hiperparámetros oculto ni se selecciona a posteriori un modelo favorable sin reportarlo.
- La consigna obliga a declarar el uso de IA y los prompts principales. Encontrarás un modelo para adaptar en `docs/DECLARACION_IA.md`.

### Documentación externa útil

- Gymnasium — [uso de reset, step y seeding](https://gymnasium.farama.org/introduction/basic_usage/)
- Gymnasium — [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- Gymnasium — [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/)
- PyTorch — [guardar y recuperar modelos `state_dict`](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- Sutton y Barto, *Reinforcement Learning: An Introduction* (2.a ed.)
- Mnih et al., *Human-level control through deep reinforcement learning* (2015)
- van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning* (2016)

> La consigna especifica el umbral **475** en CartPole y **−100** en Acrobot. Se usan esos valores para juzgar el trabajo, aunque los criterios de referencia de algunas versiones de Gymnasium puedan diferir. La versión exacta instalada queda guardada en `config.json`.
