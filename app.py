import io
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from AG_procesamiento import GeneticAlgorithm1D, build_function


# -----------------------------------------------------------
# Configuración general de la página
# -----------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Algoritmo Genético 1D",
    layout="wide",
)

st.title("Simulador de Algoritmo Genético 1D")


# -----------------------------------------------------------
# Estado de sesión para la función actual
# Esto permite que los ejemplos rápidos sí actualicen el input.
# -----------------------------------------------------------
if "func" not in st.session_state:
    st.session_state.func = "x**2 + 10*cos(2*x) + x*sin(5*x)"


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def evaluar_vector_seguro(f, x_values):
    """
    Evalúa un arreglo de x de forma segura.
    Si en algún punto la función falla o devuelve algo no finito,
    ese valor se marca como NaN para no romper la gráfica.
    """
    resultados = []

    for x in np.asarray(x_values, dtype=float):
        try:
            with np.errstate(all="ignore"):
                y = f(float(x))

            arr = np.asarray(y, dtype=float)

            # Esperamos un escalar por cada x
            if arr.size != 1:
                resultados.append(np.nan)
                continue

            valor = float(arr.reshape(-1)[0])

            if np.isfinite(valor):
                resultados.append(valor)
            else:
                resultados.append(np.nan)

        except Exception:
            resultados.append(np.nan)

    return np.array(resultados, dtype=float)


def set_example(expr):
    """
    Cambia la función actual desde los botones de ejemplo.
    Al usar on_click, Streamlit vuelve a ejecutar la app automáticamente.
    """
    st.session_state.func = expr

def format_value(value, decimals=6):
    """
    Formatea un número de forma segura para mostrarlo en métricas.
    """
    try:
        if value is None:
            return "N/A"

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:.{decimals}f}"
    except Exception:
        return "N/A"


# -----------------------------------------------------------
# Panel de ayuda
# -----------------------------------------------------------
with st.expander("ℹ️ Instrucciones de uso", expanded=False):
    st.markdown(
        """
        **Sintaxis para ecuaciones:**
        - Variable: `x`
        - Operadores: `+`, `-`, `*`, `/`, `**`
        - Funciones:
            - `sin(x)`, `cos(x)`, `tan(x)`
            - `exp(x)`, `log(x)`
            - `sqrt(x)`, `abs(x)`
        - Constante:
            - `pi`

        **Qué hace esta app:**
        - Genera una población inicial aleatoria dentro del dominio.
        - Evalúa qué tan buena es cada solución.
        - Aplica selección, crossover, mutación y elitismo.
        - Muestra la evolución generación por generación.

        **Interpretación visual:**
        - Los puntos representan la población.
        - La estrella representa el mejor valor histórico encontrado.
        - La generación 0 es la población inicial.
        """
    )


# -----------------------------------------------------------
# Sidebar: parámetros del AG
# -----------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del algoritmo")

eq_str = st.sidebar.text_input(
    "Función f(x):",
    key="func",
    help="Escribe la función a minimizar usando la variable x.",
)

x_min = st.sidebar.number_input("Límite inferior del dominio", value=-10.0)
x_max = st.sidebar.number_input("Límite superior del dominio", value=10.0)

pop_size = st.sidebar.number_input(
    "Tamaño de población",
    min_value=4,
    value=50,
    step=1,
    help="Cantidad de individuos por generación.",
)

generations = st.sidebar.number_input(
    "Cantidad de generaciones",
    min_value=1,
    value=80,
    step=1,
)

pc = st.sidebar.slider(
    "Probabilidad de crossover (pc)",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
)

pm = st.sidebar.slider(
    "Probabilidad de mutación (pm)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
)

sigma = st.sidebar.number_input(
    "Sigma relativo de mutación",
    min_value=0.001,
    max_value=1.0,
    value=0.10,
    step=0.01,
    help="Se multiplica por el ancho del dominio. Ejemplo: 0.10 equivale al 10% del rango.",
)

frame_delay = st.sidebar.slider(
    "Velocidad de animación (ms)",
    min_value=10,
    max_value=500,
    value=80,
    help="Tiempo entre generaciones en la animación.",
)

random_state = st.sidebar.number_input(
    "Semilla aleatoria (opcional)",
    min_value=0,
    value=42,
    step=1,
    help="Usa la misma semilla para repetir resultados.",
)

run_button = st.sidebar.button("▶️ Ejecutar simulación", use_container_width=True)


# -----------------------------------------------------------
# Texto principal
# -----------------------------------------------------------
with st.expander("🧬 Fundamento teórico del algoritmo genético", expanded=False):
    st.markdown("""
    Un **algoritmo genético** es un método metaheurístico de optimización inspirado en principios de evolución biológica.  
    Su propósito consiste en aproximar soluciones óptimas mediante la evolución iterativa de una población de candidatos.

    En esta aplicación se considera un problema de **optimización unidimensional**, donde cada individuo de la población
    representa un valor real $ x \in [a,b]$, y el objetivo es aproximar un minimizador de una función $ f(x) $.

    ## Esquema general del método

    ### 1. Inicialización
    Se genera una población inicial de tamaño finito dentro del dominio de búsqueda:
    $$
    x_i^{(0)} \in [a,b], \quad i=1,\dots,N
    $$
    donde $ N $ es el tamaño de la población.

    ### 2. Evaluación
    Cada individuo es evaluado mediante la función objetivo $ f(x) $.  
    Dado que el problema planteado es de **minimización**, los individuos con menor valor de $ f(x) $ se consideran más aptos.

    ### 3. Selección
    Se eligen individuos de la población actual para actuar como progenitores.  
    En esta implementación se emplea **selección por torneo**, mecanismo que favorece la reproducción de individuos con mejor desempeño relativo.

    ### 4. Crossover
    A partir de dos progenitores, se genera un descendiente mediante combinación lineal convexa:
    $$
    x_{hijo} = \\alpha x_1 + (1-\\alpha)x_2, \quad \\alpha \in [0,1]
    $$
    Este operador permite recombinar información genética de soluciones previamente encontradas.

    ### 5. Mutación
    El descendiente puede experimentar una perturbación aleatoria gaussiana:
    $$
    x' = x + \\varepsilon, \quad \\varepsilon \\sim \\mathcal{N}(0,\\sigma^2)
    $$
    La mutación introduce diversidad en la población y reduce la probabilidad de convergencia prematura hacia óptimos locales.

    ### 6. Elitismo
    El mejor individuo encontrado hasta el momento se preserva en la siguiente generación.  
    Este criterio garantiza que la calidad de la mejor solución histórica no se degrade durante la evolución.

    ### 7. Iteración generacional
    El proceso de selección, recombinación y mutación se repite durante un número prefijado de generaciones,
    produciendo una sucesión de poblaciones:
    $$
    P^{(0)}, P^{(1)}, \dots, P^{(T)}
    $$

    ## Interpretación de los parámetros

    - **Función $$ f(x) $$**
      Define el problema de optimización. La aplicación busca aproximar un valor de $ x $ que minimice dicha función.

    - **Límite inferior / límite superior del dominio**  
      Delimitan el intervalo de búsqueda:
      $$
      x \in [a,b]
      $$
      Todo individuo generado o mutado es restringido a este intervalo.

    - **Tamaño de población**  
      Número de individuos presentes en cada generación.  
      Un tamaño de población mayor incrementa la diversidad de búsqueda, aunque también eleva el costo computacional.

    - **Cantidad de generaciones**  
      Número de iteraciones evolutivas del algoritmo.  
      En general, un mayor número de generaciones permite una exploración más prolongada del espacio de búsqueda.

    - **Probabilidad de crossover $$ p_c $$**  
      Probabilidad de aplicar recombinación entre dos progenitores.  
      Valores elevados suelen favorecer la explotación de información ya presente en la población.

    - **Probabilidad de mutación $$ p_m $$**  
      Probabilidad de aplicar perturbación aleatoria a un descendiente.  
      Este parámetro regula el grado de exploración estocástica.

    - **Sigma relativo de mutación**  
      Controla la magnitud de la perturbación gaussiana.  
      En la implementación, este valor se escala con la amplitud del dominio:
      $$
      \\sigma_{real} = \\sigma_{relativo}(b-a)
      $$
      Valores pequeños inducen refinamiento local; valores grandes favorecen exploración global.

    - **Semilla aleatoria**  
      Permite reproducibilidad experimental.  
      Para una misma semilla y los mismos parámetros, la ejecución produce la misma trayectoria evolutiva.

    - **Velocidad de animación**  
      Modifica únicamente la representación visual del proceso evolutivo; no altera el comportamiento del algoritmo.

    ## Interpretación de la visualización

    - La **curva** representa la función objetivo $ f(x) $.
    - Los **puntos** representan la población en una generación dada.
    - La **estrella** señala el mejor individuo histórico encontrado hasta ese instante.
    - La gráfica de evolución complementaria muestra la sucesión del mejor valor histórico de $ f(x) $ a lo largo de las generaciones.

    ## Observación metodológica
    El algoritmo genético es un método heurístico y estocástico.  
    En consecuencia, no garantiza la obtención exacta del mínimo global en todos los casos; sin embargo,
    constituye una herramienta robusta para aproximar soluciones de alta calidad en problemas donde los métodos analíticos
    o deterministas pueden resultar difíciles de aplicar.
    """)


# -----------------------------------------------------------
# Ejemplos rápidos
# -----------------------------------------------------------
st.subheader("💡 Ejemplos rápidos")

row1 = st.columns(4)
row2 = st.columns(4)
row3 = st.columns(4)

with row1[0]:
    st.button(
        "Parábola",
        use_container_width=True,
        on_click=set_example,
        args=("x**2",),
    )

with row1[1]:
    st.button(
        "Seno",
        use_container_width=True,
        on_click=set_example,
        args=("sin(2*pi*x/3)",),
    )

with row1[2]:
    st.button(
        "Gaussiana",
        use_container_width=True,
        on_click=set_example,
        args=("exp(-x**2)",),
    )

with row1[3]:
    st.button(
        "Valor absoluto",
        use_container_width=True,
        on_click=set_example,
        args=("abs(x)",),
    )

with row2[0]:
    st.button(
        "Doble pozo",
        use_container_width=True,
        on_click=set_example,
        args=("x**2 - 1)**2",),
    )

with row2[1]:
    st.button(
        "Múltiples mínimos",
        use_container_width=True,
        on_click=set_example,
        args=("x*sin(10*x)",),
    )

with row2[2]:
    st.button(
        "Logarítmica",
        use_container_width=True,
        on_click=set_example,
        args=("log(abs(x) + 1)",),
    )

with row2[3]:
    st.button(
        "Sinusoidal",
        use_container_width=True,
        on_click=set_example,
        args=("sin(5*x)",),
    )

with row3[0]:
    st.button(
        "Pendiente con oscilación",
        use_container_width=True,
        on_click=set_example,
        args=("0.1*x + sin(3*x)",),
    )

with row3[1]:
    st.button(
        "Valles múltiples",
        use_container_width=True,
        on_click=set_example,
        args=("sin(x) + sin(10*x/3)",),
    )

with row3[2]:
    st.button(
        "Terreno irregular",
        use_container_width=True,
        on_click=set_example,
        args=("0.3*sin(8*x) + 0.7*cos(15*x)",),
    )

with row3[3]:
    st.button(
        "Escalón sigmoide",
        use_container_width=True,
        on_click=set_example,
        args=("1/(1 + exp(-10*x))",),
    )

st.code(st.session_state.func, language="python")


# -----------------------------------------------------------
# Ejecución principal
# -----------------------------------------------------------
if run_button:
    # Validación simple del dominio
    if x_min >= x_max:
        st.error("El límite inferior debe ser menor que el límite superior.")
        st.stop()

    try:
        # Construir función desde el string
        f = build_function(eq_str)

        # Muestreo de la curva de fondo
        x_plot = np.linspace(x_min, x_max, 600)
        y_plot = evaluar_vector_seguro(f, x_plot)

        valid_curve = np.isfinite(y_plot)

        if valid_curve.sum() < 2:
            raise ValueError(
                "La función no pudo evaluarse correctamente en el dominio seleccionado."
            )

        # Crear y ejecutar el algoritmo genético
        ag = GeneticAlgorithm1D(
            f=f,
            bounds=(x_min, x_max),
            pop_size=int(pop_size),
            pc=pc,
            pm=pm,
            sigma=sigma,
            generations=int(generations),
            random_state=int(random_state),
        )

        with st.spinner("Ejecutando algoritmo genético..."):
            best_x, best_f, history, best_history = ag.run()

    except Exception as e:
        st.error(f"Error al ejecutar la simulación: {e}")
        st.stop()

    # -------------------------------------------------------
    # Preparar datos para la animación
    # -------------------------------------------------------
    population_frames = []
    all_y_values = y_plot[valid_curve].tolist()

    for population in history:
        pop_y = evaluar_vector_seguro(f, population)
        valid_pop = np.isfinite(pop_y)

        population_frames.append((population, pop_y, valid_pop))

        if np.any(valid_pop):
            all_y_values.extend(pop_y[valid_pop].tolist())

    # Incluir el mejor valor final para ajustar bien el eje Y
    if best_f is not None and np.isfinite(best_f):
        all_y_values.append(float(best_f))

    if len(all_y_values) == 0:
        st.error("No se pudieron obtener valores válidos para graficar.")
        st.stop()

    y_min = float(np.min(all_y_values))
    y_max = float(np.max(all_y_values))
    y_range = max(y_max - y_min, 1e-6)

    # -------------------------------------------------------
    # Figura principal
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Curva de la función
    ax.plot(x_plot[valid_curve], y_plot[valid_curve], linewidth=2, alpha=0.75, label=f"f(x) = {eq_str}")

    # Scatter de la población actual
    population_scatter = ax.scatter([], [], s=55, zorder=4, label="Población")

    # Scatter del mejor histórico
    best_scatter = ax.scatter([], [], s=160, marker="*", zorder=5, label="Mejor histórico")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # Contenedores dinámicos de Streamlit
    animation_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_frames = len(population_frames)

    # -------------------------------------------------------
    # Animación generación por generación
    # -------------------------------------------------------
    for gen_idx, ((population, pop_y, valid_pop), (best_x_hist, best_f_hist)) in enumerate(
        zip(population_frames, best_history)
    ):
        # Actualizar población visible
        if np.any(valid_pop):
            offsets = np.column_stack((population[valid_pop], pop_y[valid_pop]))
            population_scatter.set_offsets(offsets)
        else:
            population_scatter.set_offsets(np.empty((0, 2)))

        # Actualizar mejor histórico
        if best_x_hist is not None and np.isfinite(best_f_hist):
            best_scatter.set_offsets(np.array([[best_x_hist, best_f_hist]]))
        else:
            best_scatter.set_offsets(np.empty((0, 2)))

        ax.set_title(f"Algoritmo Genético 1D - Generación {gen_idx}")

        # Convertir figura a imagen para Streamlit
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)

        # Mostrar frame
        animation_placeholder.image(buf, use_container_width=True)

        # Actualizar progreso
        progress = int(100 * (gen_idx + 1) / total_frames)
        progress_bar.progress(progress)

        # Texto de estado
        if best_x_hist is not None and np.isfinite(best_f_hist):
            status_text.text(
                f"Generación {gen_idx}/{total_frames - 1}  |  "
                f"Mejor histórico: x = {best_x_hist:.6f}, f(x) = {best_f_hist:.6f}"
            )
        else:
            status_text.text(f"Generación {gen_idx}/{total_frames - 1}")

        # Pausa para controlar velocidad de animación
        time.sleep(frame_delay / 1000)

    plt.close(fig)

    st.success("✅ Simulación completada.")
    st.caption("La generación 0 corresponde a la población inicial.")

    # -------------------------------------------------------
    # Resultados finales
    # -------------------------------------------------------
    st.subheader("📊 Resultados")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mejor x encontrado", format_value(best_x))

    with col2:
        st.metric("Mejor f(x)", format_value(best_f))

    with col3:
        st.metric("Generaciones evaluadas", str(len(history) - 1))

    # -------------------------------------------------------
    # Gráfica extra: evolución del mejor histórico
    # -------------------------------------------------------
    best_values = []
    for _, best_f_hist in best_history:
        if best_f_hist is not None and np.isfinite(best_f_hist):
            best_values.append(float(best_f_hist))
        else:
            best_values.append(np.nan)

    if len(best_values) > 0:
        st.subheader("📉 Evolución del mejor valor encontrado")

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(best_values, marker="o", linewidth=1.5)
        ax2.set_xlabel("Generación")
        ax2.set_ylabel("Mejor f(x) histórico")
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2)
        plt.close(fig2)

    if st.button("🔄 Ejecutar otra simulación", use_container_width=True):
        st.rerun()