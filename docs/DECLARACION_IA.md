# Modelo de declaración de uso de IA generativa (adaptar a lo realmente utilizado)

> Se utilizaron herramientas de IA generativa como asistentes para estructurar el código Python/PyTorch, identificar puntos de prueba, revisar documentación y redactar textos preliminares. La implementación se verificó mediante pruebas unitarias y un ensayo de integración con un entorno de prueba simplificado; los autores son responsables de ejecutar y comprobar los experimentos en Gymnasium, corregir cualquier fallo, interpretar los resultados reales y validar las referencias. No se aceptaron métricas, gráficos ni citas generadas sin validación experimental. Cualquier contribución adicional de IA deberá declararse en la versión final.

**Importante:** si se entregan gráficos obtenidos mediante `smoke` o el entorno artificial de tests como si fueran retornos de CartPole o Acrobot, la afirmación sería incorrecta. Registra siempre las corridas reales antes de rellenar el informe.

## Prompts esenciales (registrar solo los realmente utilizados)

1. "Implementa desde cero un Q-learning tabular con discretización y DQN en PyTorch para Gymnasium. Mantén comparaciones con presupuesto de pasos idéntico, evalúa sin exploración y registra tres semillas".
2. "Agrega barridos OFAT de al menos dos hiperparámetros y una ablación que retire solo la red objetivo, manteniendo el resto del pipeline".
3. "Revisa el código de Bellman, particularmente la diferencia entre terminated y truncated; escribe pruebas unitarias e indica límites de reproducibilidad".
4. "Genera gráficos únicamente a partir de CSV de entrenamiento y evaluación reales. No inventes resultados".

Comprueba en tu historial y edita esta lista para representar el uso efectivo de IA, sin atribuirle a ninguna persona validaciones que todavía no realizó.
