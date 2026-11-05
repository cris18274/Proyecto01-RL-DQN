# Auditoría frente a la consigna — 5 noviembre 2026

Auditoría de archivos: 42 corridas (12 Tier1 80k + 12 Tier1 200k + 18 Tier2) con tres semillas 11/22/33, 100 episodios finales por corrida, 5.16M pasos de entrenamiento y 4 GIF; `python -m pytest -q` pasó 9 tests de integración/algoritmos en el entorno de auditoría. No es certificación de que GitHub ya haya sido publicado.

| Criterio | Puntos | Evidencia | Estado |
|---|---:|---|---|
| Análisis solo BipedalWalker/MuJoCo/Atari | 0.5 | `report/Paper_RL_DQN_Revisado.pdf`, Sección VII, y `docs/ANALISIS_SOLO.md` | Redactado; verificar extensión exigida |
| Tier 1 | 1.0 | `results/tier1`, `results/tier1_200k`, 6/10/20 bins + DQN | Cumple experimento, no promedio umbral 475 |
| Tier 2 | 1.0 | `results/tier2`, 3 LR, 3 target cadences, sin target | `target_250` supera umbral en 3/3 seeds |
| Rigor experimental | 0.5 | Config + CSV + 3 seeds + SD + holdout por corrida | Cumple mínimo |
| Interpretación | 0.5 | Sección VI discusión | Cumple; sin causalidad universal |
| Repo Git/GitHub | 0.5 | Este ZIP sin `.venv`, README, scripts, figuras, pesos, GIF | Falta crear remoto y hacer push |
| Defensa oral | 5.0 | Presentación, script, agentes y guía de edición | Evaluación futura, ensayar |
| Paper | 1.0 | `report/Paper_RL_DQN_Revisado.pdf` a doble columna | Autor/email/URL y prompts por confirmar |

El profesor exige 1–2 páginas de análisis solo y que el paper incorpore anexo de prompts. El manuscrito revisado amplía el análisis y agrega extractos exactos disponibles; **no inventa prompts faltantes**.

**Limitaciones importantes:** (a) Tier3 requerido solo si participan dos personas; Double DQN es opcional para trabajo individual; (b) no se ejecutó estudio independiente de rangos de bins, aunque sí se compararon tres granularidades; (c) 200k cambia también calendario de exploración epsilon; (d) 3 semillas limitan significancia; (e) el estudio de ablation sin target no prueba necesidad universal cuando baseline ya falla.

**Figuras del manuscrito:** `python report/make_paper_figures.py` genera las cinco figuras a partir de los CSV originales y las guarda en `report/figures_reproduced` sin alterar las figuras entregadas.
