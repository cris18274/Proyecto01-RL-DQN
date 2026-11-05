# Evidencia verificable - Proyecto 01 RL/DQN

Contenido de esta copia: scripts del agente, configuraciones por corrida, 42 entrenamientos completos, 4,200 episodios de evaluacion final, graficas, modelos y 4 GIF. Excluye .venv y caches de Windows para facilitar su publicacion.

| Condicion | Entrenamientos | Pasos por entrenamiento | Total pasos entrenamiento | Retorno destacado |
|---|---:|---:|---:|---|
| CartPole, inicial (tier1) | 12 | 80,000 | 960,000 | Q-learning 10 bins = 276.47 +/- 101.51 (0/3 sobre 475) |
| CartPole, seguimiento (tier1_200k) | 12 | 200,000 | 2,400,000 | Q-learning 10 bins = 414.69 +/- 59.48 (1/3 sobre 475) |
| Acrobot (tier2) | 18 | 100,000 | 1,800,000 | DQN target_250 = -88.39 +/- 5.69 (3/3 sobre -100) |

- Total 42 corridas, 5,160,000 pasos de entrenamiento, 4,200 episodios en holdout final.
- Los promedios +/- desviacion estandar son entre TRES semillas de entrenamiento: 11, 22, 33.
- Cada semilla incluye una evaluacion greedy final de 100 episodios.
- No utilizar results/smoke como evidencia cientifica.
- La ampliacion 200k fue una corrida nueva; NO continuo los pesos 80k. El calendario epsilon depende del presupuesto y tambien se amplio. No aisla causalmente el efecto de incrementar pasos.
- La ablacion sin target y el DQN base fallaron en Acrobot, mientras que target_250 funciono; no constituye prueba de necesidad universal de la red objetivo.
- El reloj incluye evaluacion y no mide solamente coste de optimizacion.

## Reproducir figuras y auditoria

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\01_instalar.ps1
.\.venv\Scripts\python.exe -m rl01.cli analyze --tier-dir results/tier1
.\.venv\Scripts\python.exe -m rl01.cli analyze --tier-dir results/tier1_200k
.\.venv\Scripts\python.exe -m rl01.cli analyze --tier-dir results/tier2
```

La copia actual incluye archivos generados; NO hace falta volver a entrenar para obtener las figuras. Si se desea ejecutar otro experimento, usar un nuevo --out para evitar borrar resultados existentes.

## Configuracion reportada

Windows 11; Python 3.12.7; PyTorch 2.15.0+cpu; Gymnasium 1.4.0; numpy 2.5.4; CPU Intel x86, 12 logical CPUs; `torch.set_num_threads(1)` en el script. Verifique `config.json` para cada corrida.

## Integridad

Se verificaron las medias de 100 episodios finales contra `summary.json` para cada corrida. Los GIF son medios cualitativos, no evidencia adicional de convergencia.
