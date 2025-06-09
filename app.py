import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from numpy import sin, cos, exp, log, sqrt, pi,abs,tan
import random
# Funciones seguras
safe_dict = {
    "x": 0, "sin": sin, "cos": cos, "tan": tan,
    "exp": exp, "log": log, "sqrt": sqrt, "abs": abs,
    "pi": pi, "__builtins__": {}
}

def caminata_aleatoria_1d(eq_str, x0, x_range, pasos, tam_vec, tipo_aleatorio='discreto', dominio_completo=False):
    """
    Realiza una caminata aleatoria en 1D optimizando una función
    
    Args:
        eq_str: String de la ecuación a optimizar
        x0: Punto inicial
        x_range: Tupla con (mínimo, máximo) del dominio
        pasos: Número de pasos a realizar
        tam_vec: Tamaño de la vecindad para movimientos
        tipo_aleatorio: 'discreto' ([-1,1]) o 'continuo' (uniforme entre -1 y 1)
        dominio_completo: Si True, explora todo el dominio en lugar de vecindad local
        
    Returns:
        path: Camino seguido
        f: Función evaluadora
        f_values: Valores de la función en el camino
        min_global: Mejor valor encontrado
        paso_min_global: Paso donde se encontró el mejor valor
    """
    f = lambda x: eval(eq_str, safe_dict | {"x": x})
    path = [x0]
    f_values = [f(x0)]
    min_local = f_values[0]
    paso_min_local = 0
    
    for i in range(pasos):
        x_actual = path[-1]
        f_actual = f_values[-1]
        
        # Determinar el nuevo candidato
        if dominio_completo:
            # Explorar todo el dominio
            x_nuevo = random.uniform(x_range[0], x_range[1])
        else:
            # Explorar vecindad local
            if tipo_aleatorio == 'continuo':
                # Número aleatorio continuo entre -1 y 1
                direccion = random.uniform(-1, 1)
            else:  # Por defecto discreto
                # Elección aleatoria entre -1 y 1
                direccion = random.choice([-1, 1])
            
            x_nuevo = x_actual + direccion * tam_vec
        
        # Verificar límites y evaluar
        if x_range[0] <= x_nuevo <= x_range[1]:
            f_nuevo = f(x_nuevo)
            
            # Solo moverse si mejora
            if f_nuevo < f_actual:
                path.append(x_nuevo)
                f_values.append(f_nuevo)
                
                # Actualizar mejor valor global
                if f_nuevo < min_local:
                    min_local = f_nuevo
                    paso_min_local = i + 1
            else:
                # Mantener posición actual
                path.append(x_actual)
                f_values.append(f_actual)
        else:
            # Si se sale de los límites, quedarse en el mismo lugar
            path.append(x_actual)
            f_values.append(f_actual)
    
    return np.array(path), f, f_values, min_local, paso_min_local


# Interfaz Streamlit
st.title("Simulador de Caminata Aleatoria")
# Panel de ayuda expandible
with st.expander("ℹ️ Instrucciones de uso", expanded=False):

        st.markdown("""
        **Sintaxis para ecuaciones:**
        - Variable: `x`
        - Operadores: `+`, `-`, `*`, `/`, `**` (potencia)
        - Funciones: 
            - `sin(x)`, `cos(x)`, `tan(x)`
            - `exp(x)`, `log(x)` (natural)
            - `sqrt(x)`, `abs(x)`
        - Constantes: `pi` (3.1416)
        
        **Precauciones:**
        - Evite funciones o valores no definidos en los reales.
        
        **Parametros de busqueda**
        - **Aleatoriedad continua:** Valores entre -1 y 1.
        - **Aleatoriedad discreta:** -1 o 1 (direcciones definidas)
        - **Tamaño del paso o vecindad:** Define el tamaño de exploración de la caminata por cada iteración(Depende del tipo de aleatoridad)
        - **Dominio completo:** Saltos aleatorios en todo el rango
        
        
        """)
# Configuración en la barra lateral
st.sidebar.header("⚙️ Parámetros de simulación")
eq_str = st.sidebar.text_input("Función f(x):", "x**2", help="Ingrese la función matemática a explorar")
x_min = st.sidebar.number_input("Rango mínimo de x", value=-5.0)
x_max = st.sidebar.number_input("Rango máximo de x", value=5.0)

tipo_aleatorio = st.sidebar.radio("Tipo de aleatoriedad:", 
                                  ["discreto", "continuo"], 
                                  index=0,
                                  help="Discreto: elige entre -1 o 1. Continuo: cualquier valor entre -1 y 1")

dominio_completo = st.sidebar.checkbox("Explorar todo el dominio", 
                                      value=False,
                                      help="Si está marcado, salta a cualquier punto del dominio en lugar de moverse localmente")

# Solo mostrar tamaño de vecindad si no estamos en modo dominio completo
if not dominio_completo:
    if tipo_aleatorio=="discreto":
        tam_vec = st.sidebar.number_input("Tamaño de paso(±)", 
                                        value=0.1, 
                                        min_value=0.01, 
                                        max_value=abs(x_max - x_min))
    else:
        tam_vec = st.sidebar.number_input("Tamaño de la vecindad(±)", 
                                        value=0.1, 
                                        min_value=0.01, 
                                        max_value=abs(x_max - x_min))
else:
    tam_vec = 0.1  # Valor por defecto que no se usa pero necesario para la función

# Opción para punto inicial aleatorio
punto_inicial_aleatorio = st.sidebar.checkbox("Punto inicial aleatorio", 
                                             value=False, 
                                             help="Si se marca, la posición inicial será aleatoria dentro del rango")

if punto_inicial_aleatorio:
    x_ini = None  # Se generará aleatoriamente más adelante
    st.sidebar.info("Posición inicial se generará aleatoriamente")
else:
    x_ini = st.sidebar.slider("Posición inicial (x0)", x_min, x_max, 0.0)

pasos = st.sidebar.number_input("Cantidad de pasos", 
                               min_value=10, 
                               value=100,
                               help="Número de iteraciones de la caminata")
frame_delay = st.sidebar.slider("Velocidad de animación (ms)", 
                               10, 500, 100,
                               help="Tiempo entre frames durante la animación")

# Mensaje contextual sobre la estrategia seleccionada
if dominio_completo:
    if tipo_aleatorio=="discreto":
        st.sidebar.info("🔍 Modo: Exploración de TODO el dominio")
else:
    if tipo_aleatorio == "discreto":
        st.sidebar.info(f"🔍 Modo: Vecindad local con pasos de ±{tam_vec}")
    else:
        st.sidebar.info(f"🔍 Modo: Vecindad local con pasos entre ±{tam_vec}")

# Sección principal
if st.sidebar.button("▶️ Ejecutar simulación", use_container_width=True):
    # Generar punto inicial aleatorio si se seleccionó
    if punto_inicial_aleatorio:
        x_ini = random.uniform(x_min, x_max)
        st.info(f"Posición inicial generada aleatoriamente: x0 = {x_ini:.4f}")
    else:
        st.info(f"Posición inicial fija: x0 = {x_ini:.4f}")
    
    with st.spinner(f"Calculando caminata con {pasos} pasos..."):
        path, f, y_vals, min_global, paso_min_global = caminata_aleatoria_1d(
            eq_str, x_ini, (x_min, x_max), pasos, tam_vec
        )

    # Generar puntos para la función de fondo
    x_plot = np.linspace(x_min, x_max, 300)
    y_plot = [f(x) for x in x_plot]
    y_min, y_max = min(y_plot + y_vals), max(y_plot + y_vals)
    y_range = y_max - y_min
    
    # Configurar la figura
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_plot, y_plot, 'b-', linewidth=2, alpha=0.7, label=f"f(x) = {eq_str}")
    point = ax.scatter([], [], c='red', s=80, zorder=5, label="Posición actual")
    line, = ax.plot([], [], 'r--', alpha=0.6, linewidth=1.5, label="Trayectoria")
    #min_point = ax.scatter([], [], c='green', s=100, zorder=6, marker='*', label="Mínimo encontrado")
    
    # Configurar ejes y estilo
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    ax.set_title(f"Evolución de la Caminata Aleatoria: f(x) = {eq_str}", fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')
    
    # Contenedores para la animación y progreso
    animation_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Variables para rastrear el mejor punto
    best_x = path[0]
    best_y = y_vals[0]
    
    # Crear la animación frame por frame
    for i in range(len(path)):
        # Actualizar mejor punto encontrado
        if y_vals[i] < best_y:
            best_x = path[i]
            best_y = y_vals[i]
        
        # Actualizar datos
        line.set_data(path[:i+1], y_vals[:i+1])
        point.set_offsets([[path[i], y_vals[i]]])
        
        # Convertir figura a imagen
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        
        # Mostrar frame en Streamlit
        animation_placeholder.image(buf, use_container_width=True)
        
        # Actualizar barra de progreso y estado
        progress = int(100 * (i+1) / len(path))
        progress_bar.progress(progress)
        

        status_text.text(f"Paso {i+1}/{len(path)} - x = {path[i]:.4f}, f(x) = {y_vals[i]:.4f}")
        
        # Pausa para controlar velocidad
        time.sleep(frame_delay/1000)
    
    plt.close(fig)
    st.success("✅ Simulación completada!")
    
    # Resultados finales
    st.subheader("📊 Resultados")
    
    # Crear columnas para métricas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Posición inicial(x)", f"{path[0]:.4f}")
    with col2:
        st.metric("Posición final(x)", f"{path[-1]:.4f}")
    with col3:
        st.metric("Menor valor encontrado para f(x)", f"{min_global:.4f}", f"en paso {paso_min_global}")
    
    # Botón para nueva simulación
    if st.button("🔄 Realizar nueva simulación", use_container_width=True):
        st.experimental_rerun()
# Mensaje inicial
else:
    st.markdown("""
    Visualiza cómo una partícula explora una función matemática mediante pasos aleatorios,
    aceptando solo movimientos que mejoran su posición (reducen el valor de la función).
    
    **Para comenzar:**
    1. Ingresa una función matemática en la barra lateral
    2. Configura los parámetros de simulación
    3. Haz clic en 'Ejecutar simulación'
    
    ### Características:
    - **Punto inicial:** Fijo o aleatorio dentro del rango
    - **Animación en tiempo real:** Visualiza el proceso paso a paso
    - **Resultados detallados:** Muestra posición inicial, final y mejor valor encontrado
    
    ### Instrucciones para ecuaciones:
    - Usa `x` como variable independiente
    - Funciones disponibles: `sin`, `cos`, `exp`, `log`, `sqrt`
    - Ejemplos: `x**2`, `sin(2*pi*x/3)`, `exp(-x**2)`
    """)
    
    # Ejemplos interactivos
    st.subheader("💡 Ejemplos rápidos")
    cols1 = st.columns(4)
    cols2 = st.columns(4)
    cols3 = st.columns(4)

    # Primera fila de botones
    with cols1[0]:
        if st.button("Parábola", use_container_width=True):
            st.session_state.func = "x**2"
    with cols1[1]:
        if st.button("Seno", use_container_width=True):
            st.session_state.func = "sin(2*pi*x/3)"
    with cols1[2]:
        if st.button("Gaussiana", use_container_width=True):
            st.session_state.func = "exp(-x**2)"
    with cols1[3]:
        if st.button("Valor absoluto", use_container_width=True):
            st.session_state.func = "abs(x)"

    # Segunda fila de botones
    with cols2[0]:
        if st.button("Doble pozo", use_container_width=True):
            st.session_state.func = "(x**2 - 1)**2"
    with cols2[1]:
        if st.button("Múltiples mínimos", use_container_width=True):
            st.session_state.func = "x*sin(10*x)"
    with cols2[2]:
        if st.button("Logarítmica", use_container_width=True):
            st.session_state.func = "log(abs(x) + 1)"
    with cols2[3]:
        if st.button("Sinusoidal", use_container_width=True):
            st.session_state.func = "sin(5*x)"

    # Tercera fila de botones
    with cols3[0]:
        if st.button("Pendiente con oscil", use_container_width=True):
            st.session_state.func = "0.1*x + sin(3*x)"
    with cols3[1]:
        if st.button("Valles múltiples", use_container_width=True):
            st.session_state.func = "sin(x) + sin(10*x/3)"
    with cols3[2]:
        if st.button("Terreno irregular", use_container_width=True):
            st.session_state.func = "0.3*sin(8*x) + 0.7*cos(15*x)"
    with cols3[3]:
        if st.button("Escalón", use_container_width=True):
            st.session_state.func = "1/(1 + exp(-10*x))"

    # Mostrar la función actual si existe en session_state
    if hasattr(st.session_state, 'func'):
        st.code(f"{st.session_state.func}")