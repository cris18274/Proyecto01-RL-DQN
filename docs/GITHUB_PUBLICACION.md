# Publicación correcta en GitHub desde Windows PowerShell

El profesor pide un repositorio Git. Este paquete está **preparado para publicarlo**, pero **no se ha publicado** en ninguna cuenta: falta autenticarte y hacer `git push`.

1. Crea un repositorio NUEVO y vacío en https://github.com/new (por ejemplo `Proyecto01-RL-DQN`). No marques opciones para añadir README, .gitignore o licencia, pues ya existen archivos locales.
2. Descomprime el ZIP GitHub listo y abre PowerShell dentro de la carpeta `Proyecto01_RL_DQN` que contiene `README.md` y `rl01`.
3. Ejecuta:

```powershell
git --version
git init
git branch -M main
git add .
git status --short
git commit -m "Entrega Proyecto 01: Q-learning y DQN"
git remote add origin https://github.com/TU_USUARIO/Proyecto01-RL-DQN.git
git remote -v
git push -u origin main
```

**Sustituye `TU_USUARIO` por tu usuario real.** Si Git te pide identificar el autor, establece tu nombre/correo de Git; no compartas contraseñas ni tokens en chats o capturas. GitHub puede abrir un navegador para autenticación.

4. Copia la URL real y colócala en el PDF/paper y la presentación antes de entregar. Abre ese enlace en otro navegador o en modo privado para comprobar permisos.

5. Si utilizas repositorio **privado**, concede acceso al docente; si es público, revisa que no haya datos personales o secretos antes del push.

### ¿Qué subir / qué NO subir?

- **Sí:** `rl01/`, `tests/`, `windows/`, `docs/`, `README.md`, `requirements*.txt`, `results/` (42 corridas, figuras, pesos, cuatro GIF), `report/`, `slides/`.
- **No:** `.venv/` (249 MB en la copia original), `__pycache__/`, `.pytest_cache/`, claves, tokens, archivos privados, zip bruto original.

### Si aparece un error

- `git: command not found`: instala Git for Windows desde git-scm.com, luego abre nueva terminal.
- `remote origin already exists`: revisa primero `git remote -v`; usa `git remote set-url origin URL` solo si la URL previa es equivocada.
- `rejected non-fast-forward`: puede ocurrir si inicializaste el repositorio remoto con un README; no fuerces el push: revisa y fusiona historias de manera consciente.
- `src refspec main does not match any`: falta el primer `git commit` o la rama se llama distinto.

Verificación final:

```powershell
git status
git log -1 --oneline
git remote -v
```

En la web de GitHub confirma presencia de `report/Paper_RL_DQN_Revisado.pdf`, `slides/Defensa_RL_DQN_Resultados_Reales.pptx`, `results/tier2/target_250_seed0011/summary.json`, `rl01/dqn.py` y `README.md`.
