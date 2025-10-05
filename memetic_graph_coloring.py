"""
Algoritmo Memético para Coloración de Grafos
============================================

Implementación completa de un algoritmo memético que combina algoritmos genéticos
con búsqueda local para resolver el problema de coloración de grafos.

Autor: GitHub Copilot
Fecha: Septiembre 2025
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time
from collections import defaultdict


class Graph:
    """
    Clase para representar un grafo y manejar operaciones relacionadas
    con la coloración de grafos.
    """
    
    def __init__(self, num_vertices: int = 0):
        self.num_vertices = num_vertices
        self.adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        self.edges = []
        
    def load_from_file(self, filepath: str) -> None:
        """
        Carga un grafo desde archivo en formato DIMACS.
        
        Args:
            filepath: Ruta al archivo del grafo
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line.startswith('p edge'):
                    # Línea de problema: p edge vertices edges
                    parts = line.split()
                    self.num_vertices = int(parts[2])
                    self.adjacency_matrix = np.zeros((self.num_vertices, self.num_vertices), dtype=int)
                elif line.startswith('e'):
                    # Línea de arista: e vertex1 vertex2
                    parts = line.split()
                    v1, v2 = int(parts[1]) - 1, int(parts[2]) - 1  # Convertir a índice 0
                    self.add_edge(v1, v2)
                    
        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {filepath}")
        except Exception as e:
            print(f"Error al cargar el grafo: {e}")
    
    def add_edge(self, v1: int, v2: int) -> None:
        """Añade una arista entre dos vértices."""
        if 0 <= v1 < self.num_vertices and 0 <= v2 < self.num_vertices:
            self.adjacency_matrix[v1][v2] = 1
            self.adjacency_matrix[v2][v1] = 1
            self.edges.append((v1, v2))
    
    def get_neighbors(self, vertex: int) -> List[int]:
        """Retorna lista de vecinos de un vértice."""
        return [i for i in range(self.num_vertices) if self.adjacency_matrix[vertex][i] == 1]
    
    def is_valid_coloring(self, coloring: List[int]) -> bool:
        """
        Verifica si una coloración es válida (sin vértices adyacentes del mismo color).
        
        Args:
            coloring: Lista con el color asignado a cada vértice
            
        Returns:
            True si la coloración es válida, False en caso contrario
        """
        if len(coloring) != self.num_vertices:
            return False
            
        for v1, v2 in self.edges:
            if coloring[v1] == coloring[v2]:
                return False
        return True
    
    def count_conflicts(self, coloring: List[int]) -> int:
        """
        Cuenta el número de conflictos (aristas con vértices del mismo color).
        
        Args:
            coloring: Lista con el color asignado a cada vértice
            
        Returns:
            Número de conflictos
        """
        conflicts = 0
        for v1, v2 in self.edges:
            if coloring[v1] == coloring[v2]:
                conflicts += 1
        return conflicts
    
    def get_chromatic_number_upper_bound(self) -> int:
        """Calcula una cota superior del número cromático usando algoritmo greedy."""
        coloring = [-1] * self.num_vertices
        
        for vertex in range(self.num_vertices):
            # Encontrar colores usados por vecinos
            used_colors = set()
            for neighbor in self.get_neighbors(vertex):
                if coloring[neighbor] != -1:
                    used_colors.add(coloring[neighbor])
            
            # Asignar el menor color disponible
            color = 0
            while color in used_colors:
                color += 1
            coloring[vertex] = color
        
        return max(coloring) + 1 if coloring else 0


class Individual:
    """
    Representa un individuo en la población del algoritmo genético.
    Cada individuo es una posible coloración del grafo.
    """
    
    def __init__(self, graph: Graph, num_colors: int, coloring: Optional[List[int]] = None):
        self.graph = graph
        self.num_colors = num_colors
        
        if coloring is None:
            # Generar coloración aleatoria
            self.coloring = [random.randint(0, num_colors - 1) for _ in range(graph.num_vertices)]
        else:
            self.coloring = coloring.copy()
        
        self.fitness = self._calculate_fitness()
    
    def _calculate_fitness(self) -> float:
        """
        Calcula la fitness del individuo.
        Fitness = 1 / (1 + número_de_conflictos)
        """
        conflicts = self.graph.count_conflicts(self.coloring)
        return 1.0 / (1.0 + conflicts)
    
    def mutate(self, mutation_rate: float) -> None:
        """
        Aplica mutación al individuo cambiando aleatoriamente algunos colores.
        
        Args:
            mutation_rate: Probabilidad de mutación para cada gen
        """
        for i in range(len(self.coloring)):
            if random.random() < mutation_rate:
                self.coloring[i] = random.randint(0, self.num_colors - 1)
        self.fitness = self._calculate_fitness()
    
    def local_search(self, max_iterations: int = 100) -> None:
        """
        Aplica búsqueda local para mejorar la solución.
        Intenta reducir conflictos cambiando colores de vértices conflictivos.
        
        Args:
            max_iterations: Número máximo de iteraciones de búsqueda local
        """
        for _ in range(max_iterations):
            conflicts = self._get_conflicted_vertices()
            if not conflicts:
                break  # No hay conflictos, solución óptima local
            
            # Seleccionar vértice conflictivo aleatoriamente
            vertex = random.choice(conflicts)
            
            # Intentar cambiar a un color que reduzca conflictos
            best_color = self.coloring[vertex]
            best_conflicts = self._count_vertex_conflicts(vertex, self.coloring[vertex])
            
            for color in range(self.num_colors):
                if color != self.coloring[vertex]:
                    vertex_conflicts = self._count_vertex_conflicts(vertex, color)
                    if vertex_conflicts < best_conflicts:
                        best_color = color
                        best_conflicts = vertex_conflicts
            
            if best_color != self.coloring[vertex]:
                self.coloring[vertex] = best_color
                self.fitness = self._calculate_fitness()
    
    def _get_conflicted_vertices(self) -> List[int]:
        """Retorna lista de vértices que tienen conflictos."""
        conflicted = []
        for vertex in range(self.graph.num_vertices):
            if self._count_vertex_conflicts(vertex, self.coloring[vertex]) > 0:
                conflicted.append(vertex)
        return conflicted
    
    def _count_vertex_conflicts(self, vertex: int, color: int) -> int:
        """Cuenta conflictos de un vértice si tuviera cierto color."""
        conflicts = 0
        for neighbor in self.graph.get_neighbors(vertex):
            if self.coloring[neighbor] == color:
                conflicts += 1
        return conflicts
    
    def copy(self) -> 'Individual':
        """Crea una copia del individuo."""
        return Individual(self.graph, self.num_colors, self.coloring)


class MemeticAlgorithm:
    """
    Implementación del Algoritmo Memético para Coloración de Grafos.
    
    Combina algoritmos genéticos con búsqueda local para encontrar
    coloraciones válidas con el mínimo número de colores posible.
    """
    
    def __init__(self, graph: Graph, population_size: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 local_search_prob: float = 0.3):
        """
        Inicializa el algoritmo memético.
        
        Args:
            graph: Grafo a colorear
            population_size: Tamaño de la población
            mutation_rate: Tasa de mutación
            crossover_rate: Tasa de cruzamiento
            local_search_prob: Probabilidad de aplicar búsqueda local
        """
        self.graph = graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.local_search_prob = local_search_prob
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.generation = 0
    
    def solve(self, max_colors: int, max_generations: int = 1000, 
              target_fitness: float = 1.0) -> Tuple[List[int], int]:
        """
        Ejecuta el algoritmo memético para encontrar una coloración.
        
        Args:
            max_colors: Número máximo de colores a usar
            max_generations: Número máximo de generaciones
            target_fitness: Fitness objetivo (1.0 = sin conflictos)
            
        Returns:
            Tupla con (mejor_coloración, número_de_conflictos)
        """
        print(f"🎨 Iniciando Algoritmo Memético para coloración de grafos")
        print(f"📊 Parámetros: {self.graph.num_vertices} vértices, {len(self.graph.edges)} aristas")
        print(f"🎯 Objetivo: ≤ {max_colors} colores, {max_generations} generaciones máx.")
        print("-" * 60)
        
        start_time = time.time()
        
        # Inicializar población
        self._initialize_population(max_colors)
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluar población
            self._evaluate_population()
            
            # Aplicar búsqueda local a algunos individuos
            self._apply_local_search()
            
            # Verificar criterio de parada
            if self.best_individual.fitness >= target_fitness:
                conflicts = self.graph.count_conflicts(self.best_individual.coloring)
                elapsed_time = time.time() - start_time
                print(f"✅ ¡Solución encontrada en generación {generation}!")
                print(f"⏱️  Tiempo: {elapsed_time:.2f}s")
                print(f"🎯 Conflictos: {conflicts}")
                return self.best_individual.coloring, conflicts
            
            # Crear nueva generación
            new_population = self._create_new_generation()
            self.population = new_population
            
            # Mostrar progreso cada 100 generaciones
            if generation % 100 == 0:
                conflicts = self.graph.count_conflicts(self.best_individual.coloring)
                print(f"Gen {generation:4d}: Fitness = {self.best_individual.fitness:.4f}, "
                      f"Conflictos = {conflicts}")
        
        # Algoritmo terminado
        conflicts = self.graph.count_conflicts(self.best_individual.coloring)
        elapsed_time = time.time() - start_time
        print(f"🔄 Algoritmo terminado después de {max_generations} generaciones")
        print(f"⏱️  Tiempo total: {elapsed_time:.2f}s")
        print(f"🎯 Mejor solución: {conflicts} conflictos")
        
        return self.best_individual.coloring, conflicts
    
    def _initialize_population(self, num_colors: int) -> None:
        """Inicializa la población con individuos aleatorios."""
        self.population = []
        for _ in range(self.population_size):
            individual = Individual(self.graph, num_colors)
            self.population.append(individual)
    
    def _evaluate_population(self) -> None:
        """Evalúa la población y actualiza el mejor individuo."""
        # Ordenar por fitness (descendente)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Actualizar mejor individuo
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = self.population[0].copy()
        
        # Guardar estadísticas
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        self.fitness_history.append(avg_fitness)
    
    def _apply_local_search(self) -> None:
        """Aplica búsqueda local a algunos individuos seleccionados."""
        for individual in self.population:
            if random.random() < self.local_search_prob:
                individual.local_search(max_iterations=50)
    
    def _create_new_generation(self) -> List[Individual]:
        """Crea una nueva generación usando selección, cruzamiento y mutación."""
        new_population = []
        
        # Elitismo: mantener mejores individuos
        elite_size = max(1, self.population_size // 10)
        new_population.extend([ind.copy() for ind in self.population[:elite_size]])
        
        # Generar resto de la población
        while len(new_population) < self.population_size:
            # Selección de padres
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Cruzamiento
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutación
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 5) -> Individual:
        """Selección por torneo."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Cruzamiento de un punto para coloración de grafos.
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            
        Returns:
            Tupla con dos hijos
        """
        crossover_point = random.randint(1, self.graph.num_vertices - 1)
        
        child1_coloring = (parent1.coloring[:crossover_point] + 
                          parent2.coloring[crossover_point:])
        child2_coloring = (parent2.coloring[:crossover_point] + 
                          parent1.coloring[crossover_point:])
        
        child1 = Individual(self.graph, parent1.num_colors, child1_coloring)
        child2 = Individual(self.graph, parent2.num_colors, child2_coloring)
        
        return child1, child2
    
    def plot_fitness_evolution(self) -> None:
        """Grafica la evolución de la fitness a lo largo de las generaciones."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', linewidth=2)
        plt.title('Evolución de la Fitness Promedio')
        plt.xlabel('Generación')
        plt.ylabel('Fitness Promedio')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_statistics(self) -> dict:
        """Retorna estadísticas del algoritmo."""
        conflicts = self.graph.count_conflicts(self.best_individual.coloring)
        colors_used = len(set(self.best_individual.coloring))
        
        return {
            'generations': self.generation,
            'best_fitness': self.best_individual.fitness,
            'conflicts': conflicts,
            'colors_used': colors_used,
            'is_valid': conflicts == 0,
            'vertices': self.graph.num_vertices,
            'edges': len(self.graph.edges)
        }


def create_random_graph(num_vertices: int, edge_probability: float = 0.3) -> Graph:
    """
    Crea un grafo aleatorio para pruebas.
    
    Args:
        num_vertices: Número de vértices
        edge_probability: Probabilidad de que exista una arista entre dos vértices
        
    Returns:
        Grafo aleatorio
    """
    graph = Graph(num_vertices)
    
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                graph.add_edge(i, j)
    
    return graph


def main():
    """Función principal de demostración."""
    print("🎨 ALGORITMO MEMÉTICO PARA COLORACIÓN DE GRAFOS 🎨")
    print("=" * 60)
    
    # Crear grafo de prueba
    print("📊 Creando grafo de prueba...")
    graph = create_random_graph(20, 0.3)
    
    # Estimar cota superior
    upper_bound = graph.get_chromatic_number_upper_bound()
    print(f"📈 Cota superior estimada: {upper_bound} colores")
    
    # Configurar algoritmo
    ma = MemeticAlgorithm(
        graph=graph,
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        local_search_prob=0.3
    )
    
    # Ejecutar algoritmo
    coloring, conflicts = ma.solve(
        max_colors=upper_bound,
        max_generations=500,
        target_fitness=1.0
    )
    
    # Mostrar resultados
    print("\n📋 RESULTADOS FINALES:")
    print("=" * 30)
    stats = ma.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎨 Coloración final: {coloring[:10]}..." if len(coloring) > 10 else f"\n🎨 Coloración final: {coloring}")
    
    # Verificar solución
    if graph.is_valid_coloring(coloring):
        print("✅ ¡La coloración es VÁLIDA!")
    else:
        print("❌ La coloración tiene conflictos")
    
    # Graficar evolución
    ma.plot_fitness_evolution()


if __name__ == "__main__":
    main()