import pandas as pd
import numpy as np
import sympy as sp

class GeneticAlgorithm1D:
    """
    Algoritmo Genético para minimizar una función f(x)
    donde x es una variable real en un dominio [a, b].
    """

    def __init__(self, f, bounds: tuple, pop_size=30,
                 pc=0.8, pm=0.1, sigma=0.1, generations=200):
        """
        Inicializa el algoritmo genético.

        f           : función objetivo a minimizar
        bounds      : tupla (a, b) con el dominio de búsqueda
        pop_size    : tamaño de la población
        pc          : probabilidad de crossover
        pm          : probabilidad de mutación
        sigma       : tamaño relativo de la mutación
        generations : número de generaciones
        """

        # Función objetivo
        self.f = f

        # Límites del dominio
        self.a, self.b = bounds

        # Parámetros del AG
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm

        # Sigma se escala al tamaño del dominio
        # Esto evita mutaciones demasiado grandes o pequeñas
        self.sigma = sigma * (self.b - self.a)

        # Número de generaciones
        self.generations = generations

        # Estado interno del algoritmo
        self.population = None      # población actual
        self.fitness = None         # fitness de cada individuo
        self.best_x = None          # mejor solución encontrada
        self.best_f = np.inf        # mejor valor de la función

    def initialize_population(self):
        """
        Crea la población inicial.
        Cada individuo es un número real uniforme en [a, b].
        """
        self.population = np.random.uniform(
            self.a, self.b, self.pop_size
        )

    def evaluate(self):
        """
        Evalúa la población:
        - Calcula f(x) para cada individuo
        - Convierte a fitness (maximización)
        - Actualiza el mejor individuo global
        """

        # Evaluar la función objetivo para toda la población
        values = np.array([self.f(x) for x in self.population])

        # Convertimos minimización en maximización
        self.fitness = -values

        # Índice del mejor individuo de la generación actual
        idx = np.argmin(values)

        # Si es mejor que el mejor histórico, lo guardamos
        if values[idx] < self.best_f:
            self.best_f = values[idx]
            self.best_x = self.population[idx]

    def select_parents(self, k=3):
        """
        Selección por torneo.
        Se eligen k individuos al azar y se queda el mejor.
        """

        # Elegir k índices distintos al azar
        idx = np.random.choice(self.pop_size, k, replace=False)

        # De esos k, devolver el que tenga mejor fitness
        return self.population[idx[np.argmax(self.fitness[idx])]]

    def crossover(self, p1, p2):
        """
        Crossover para variables reales.
        Combina dos padres mediante una combinación convexa.
        """

        # Con probabilidad pc se hace crossover
        if np.random.rand() < self.pc:
            alpha = np.random.rand()
            return alpha * p1 + (1 - alpha) * p2

        # Si no hay crossover, el padre pasa directo
        return p1

    def mutate(self, x):
        """
        Mutación gaussiana.
        Introduce exploración aleatoria controlada.
        """

        # Con probabilidad pm se muta
        if np.random.rand() < self.pm:
            x += np.random.normal(0, self.sigma)

        # Asegura que x esté dentro del dominio
        return np.clip(x, self.a, self.b)

    def step(self):
        """
        Ejecuta UNA generación del algoritmo genético.
        """

        # Nueva población con elitismo:
        # el mejor individuo pasa directamente
        new_pop = [self.best_x]

        # Generar el resto de la población
        while len(new_pop) < self.pop_size:
            # Seleccionar padres
            p1 = self.select_parents()
            p2 = self.select_parents()

            # Crear descendiente
            child = self.crossover(p1, p2)
            child = self.mutate(child)

            # Añadir a la nueva población
            new_pop.append(child)

        # Reemplazo generacional completo
        self.population = np.array(new_pop)

    def run(self):
        """
        Ejecuta el algoritmo completo.
        """

        # Inicializar población
        self.initialize_population()

        # Ciclo evolutivo
        for _ in range(self.generations):
            self.evaluate()  # evaluar población
            self.step()      # evolucionar población

            # Enfriamiento del tamaño de mutación
            self.sigma *= 0.99

        # Devuelve la mejor solución encontrada
        return self.best_x, self.best_f

def build_function(expr_str):
    x = sp.symbols('x')

    # Parsear expresión
    expr = sp.sympify(expr_str)

    # Convertir a función numérica
    f = sp.lambdify(x, expr, modules=["numpy"])

    return f

#expr_str = "x**2 + 5*sin(3*x)"
expr_str = "x**2 + 10*cos(2*x) + x*sin(5*x)"
f = build_function(expr_str)

print(f(1.0))  # funciona

ag = GeneticAlgorithm1D(f, bounds=(-10, 10),
                          pop_size=50, generations=100)
ag.initialize_population()
#print("Población inicial:", ag.population)
evaluation = ag.evaluate()
print(f"Mejor x: {ag.best_x}, f(x): {ag.best_f}")
ag.step()
print(f"Mejor x: {ag.best_x}, f(x): {ag.best_f}")
ag.run()
print(f"Mejor x: {ag.best_x}, f(x): {ag.best_f}")

