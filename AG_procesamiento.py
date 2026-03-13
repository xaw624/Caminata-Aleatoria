import numpy as np
import sympy as sp


def _scalar_to_float_or_inf(value):
    """
    Convierte un resultado cualquiera a float.
    Si no se puede convertir o no es finito, devuelve +inf.
    Esto evita que el AG se rompa con NaN, inf o errores de dominio.
    """
    try:
        arr = np.asarray(value, dtype=float)

        # Nos aseguramos de que sea un único escalar
        if arr.size != 1:
            return np.inf

        scalar = float(arr.reshape(-1)[0])

        # Si no es finito, se castiga como infinito
        if not np.isfinite(scalar):
            return np.inf

        return scalar
    except Exception:
        return np.inf


class GeneticAlgorithm1D:
    """
    Algoritmo Genético 1D para minimizar una función f(x)
    en un dominio continuo [a, b].
    """

    def __init__(
        self,
        f,
        bounds: tuple,
        pop_size=30,
        pc=0.8,
        pm=0.1,
        sigma=0.1,
        generations=200,
        tournament_k=3,
        sigma_decay=0.99,
        random_state=None,
    ):
        """
        Parámetros:
        - f: función objetivo a minimizar
        - bounds: tupla (a, b) con el dominio de búsqueda
        - pop_size: tamaño de la población
        - pc: probabilidad de crossover
        - pm: probabilidad de mutación
        - sigma: tamaño relativo de la mutación respecto al dominio
        - generations: cantidad de generaciones
        - tournament_k: tamaño del torneo para selección
        - sigma_decay: factor de enfriamiento de sigma por generación
        - random_state: semilla opcional para reproducibilidad
        """

        # Función objetivo
        self.f = f

        # Límites del dominio
        self.a, self.b = bounds

        # Validación básica del dominio
        if self.a >= self.b:
            raise ValueError("El límite inferior debe ser menor que el superior.")

        # Validación del tamaño de población
        if pop_size < 2:
            raise ValueError("La población debe tener al menos 2 individuos.")

        # Parámetros principales del algoritmo
        self.pop_size = int(pop_size)
        self.pc = float(pc)
        self.pm = float(pm)
        self.generations = int(generations)

        # Selección por torneo
        self.tournament_k = int(max(2, min(tournament_k, self.pop_size)))

        # Sigma relativo al tamaño del dominio
        domain_size = self.b - self.a
        self.initial_sigma = float(sigma) * domain_size
        self.sigma = self.initial_sigma
        self.sigma_decay = float(sigma_decay)

        # Generador aleatorio moderno de NumPy
        self.rng = np.random.default_rng(random_state)

        # Estado interno
        self.population = None
        self.fitness = None
        self.best_x = None
        self.best_f = np.inf

        # Historial para visualización
        self.history = []        # Guarda la población de cada generación
        self.best_history = []   # Guarda (best_x, best_f) por generación

    def initialize_population(self):
        """
        Crea la población inicial uniforme dentro del dominio [a, b].
        """
        self.population = self.rng.uniform(self.a, self.b, self.pop_size)

    def evaluate_individual(self, x):
        """
        Evalúa un solo individuo de forma segura.
        Si la función falla o devuelve algo inválido, se castiga con +inf.
        """
        try:
            with np.errstate(all="ignore"):
                value = self.f(float(x))
            return _scalar_to_float_or_inf(value)
        except Exception:
            return np.inf

    def evaluate(self):
        """
        Evalúa toda la población actual:
        - Calcula f(x) para cada individuo
        - Convierte minimización a fitness de maximización usando -f(x)
        - Actualiza el mejor individuo histórico
        - Guarda historial para animación
        """
        values = np.array([self.evaluate_individual(x) for x in self.population], dtype=float) # type: ignore

        # Fitness: como el AG selecciona "mayor fitness", usamos -f(x)
        self.fitness = -values

        # Mejor individuo de la generación actual
        idx = np.argmin(values)
        gen_best_x = float(self.population[idx]) # type: ignore
        gen_best_f = float(values[idx])

        # Actualizar mejor histórico
        if gen_best_f < self.best_f:
            self.best_f = gen_best_f
            self.best_x = gen_best_x

        # Guardar historial
        self.history.append(self.population.copy()) # type: ignore
        self.best_history.append((self.best_x, self.best_f))

    def select_parent(self):
        """
        Selección por torneo.
        Elige k individuos al azar y retorna el mejor.
        """
        idx = self.rng.choice(self.pop_size, size=self.tournament_k, replace=False)
        best_local_idx = idx[np.argmax(self.fitness[idx])] # type: ignore
        return float(self.population[best_local_idx]) # type: ignore

    def crossover(self, p1, p2):
        """
        Crossover para variables reales.
        Genera un hijo mediante combinación convexa de dos padres.
        """
        if self.rng.random() < self.pc:
            alpha = self.rng.random()
            return alpha * p1 + (1 - alpha) * p2

        # Si no hay crossover, uno de los padres pasa directo
        return p1

    def mutate(self, x):
        """
        Mutación gaussiana con recorte al dominio.
        """
        if self.rng.random() < self.pm:
            x = x + self.rng.normal(0, self.sigma)

        # Mantener siempre al individuo dentro del dominio
        return float(np.clip(x, self.a, self.b))

    def step(self):
        """
        Ejecuta una generación:
        - Mantiene elitismo: el mejor histórico pasa directo
        - Genera el resto con selección, crossover y mutación
        """
        if self.best_x is None:
            raise RuntimeError("Debes evaluar la población antes de llamar a step().")

        # Elitismo: conservar al mejor histórico
        new_population = [self.best_x]

        # Completar la nueva población
        while len(new_population) < self.pop_size:
            p1 = self.select_parent()
            p2 = self.select_parent()

            child = self.crossover(p1, p2)
            child = self.mutate(child)

            new_population.append(child)

        self.population = np.array(new_population, dtype=float)

    def run(self):
        """
        Ejecuta el algoritmo completo y devuelve:
        - best_x: mejor x encontrado
        - best_f: mejor valor f(x)
        - history: historial de poblaciones
        - best_history: historial del mejor histórico por generación

        Nota:
        Se guarda también la generación 0 (población inicial).
        """
        # Reiniciar estado por si el objeto se reutiliza
        self.population = None
        self.fitness = None
        self.best_x = None
        self.best_f = np.inf
        self.history = []
        self.best_history = []
        self.sigma = self.initial_sigma

        # 1) Población inicial
        self.initialize_population()

        # 2) Evaluar generación 0
        self.evaluate()

        # 3) Evolucionar
        for _ in range(self.generations):
            self.step()
            self.sigma *= self.sigma_decay
            self.evaluate()

        return self.best_x, self.best_f, self.history, self.best_history


def build_function(expr_str):
    """
    Construye una función numérica f(x) a partir de un string usando SymPy.

    Funciones permitidas:
    sin, cos, tan, exp, log, sqrt, abs, pi
    Variable permitida:
    x
    """
    x = sp.symbols("x")

    allowed_locals = {
        "x": x,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "pi": sp.pi,
    }

    try:
        expr = sp.sympify(expr_str, locals=allowed_locals)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la expresión: {e}")

    # Validar que solo exista la variable x
    invalid_symbols = expr.free_symbols - {x}
    if invalid_symbols:
        raise ValueError(
            f"La expresión contiene variables no permitidas: {invalid_symbols}. Usa solo 'x'."
        )

    try:
        raw_f = sp.lambdify(x, expr, modules=["numpy"])
    except Exception as e:
        raise ValueError(f"No se pudo convertir la expresión a función numérica: {e}")

    def f(x_value):
        with np.errstate(all="ignore"):
            return raw_f(x_value)

    return f


# Bloque de prueba local.
# NO se ejecuta cuando este archivo se importa desde app.py
if __name__ == "__main__":
    expr_str = "x**2 + 10*cos(2*x) + x*sin(5*x)"
    f = build_function(expr_str)

    ag = GeneticAlgorithm1D(
        f=f,
        bounds=(-10, 10),
        pop_size=50,
        pc=0.8,
        pm=0.1,
        sigma=0.1,
        generations=100,
        random_state=42,
    )

    best_x, best_f, history, best_history = ag.run()
    print(f"Mejor x: {best_x}")
    print(f"Mejor f(x): {best_f}")
    print(f"Generaciones registradas: {len(history)}")

