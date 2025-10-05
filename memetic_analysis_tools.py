"""
Utilidades adicionales para análisis del Algoritmo Memético
===========================================================

Este archivo contiene herramientas adicionales para benchmarking,
análisis estadístico y comparación con otros algoritmos.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time
import statistics
from memetic_graph_coloring import Graph, MemeticAlgorithm, create_random_graph


class BenchmarkSuite:
    """Suite de benchmarks para evaluar el algoritmo memético."""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(self, graph_sizes: List[int], 
                     densities: List[float], 
                     runs_per_config: int = 5) -> Dict:
        """
        Ejecuta benchmark completo con diferentes tamaños y densidades de grafos.
        
        Args:
            graph_sizes: Lista de tamaños de grafos a probar
            densities: Lista de densidades de grafos a probar
            runs_per_config: Número de ejecuciones por configuración
            
        Returns:
            Diccionario con resultados del benchmark
        """
        print("🚀 INICIANDO BENCHMARK COMPLETO")
        print("=" * 50)
        
        results = {
            'configurations': [],
            'execution_times': [],
            'colors_used': [],
            'conflicts': [],
            'success_rate': []
        }
        
        total_configs = len(graph_sizes) * len(densities)
        current_config = 0
        
        for size in graph_sizes:
            for density in densities:
                current_config += 1
                print(f"\n📊 Configuración {current_config}/{total_configs}: "
                      f"{size} vértices, densidad {density}")
                
                config_results = self._run_single_config(size, density, runs_per_config)
                
                results['configurations'].append(f"{size}v-{density}d")
                results['execution_times'].append(config_results['avg_time'])
                results['colors_used'].append(config_results['avg_colors'])
                results['conflicts'].append(config_results['avg_conflicts'])
                results['success_rate'].append(config_results['success_rate'])
                
                print(f"   ✅ Promedio: {config_results['avg_colors']:.1f} colores, "
                      f"{config_results['avg_time']:.2f}s, "
                      f"{config_results['success_rate']*100:.1f}% éxito")
        
        self.results = results
        return results
    
    def _run_single_config(self, size: int, density: float, runs: int) -> Dict:
        """Ejecuta múltiples runs para una configuración específica."""
        times = []
        colors = []
        conflicts = []
        successes = 0
        
        for run in range(runs):
            # Crear grafo
            graph = create_random_graph(size, density)
            upper_bound = graph.get_chromatic_number_upper_bound()
            
            # Configurar algoritmo
            ma = MemeticAlgorithm(
                graph=graph,
                population_size=min(100, size * 2),
                mutation_rate=0.1,
                crossover_rate=0.8,
                local_search_prob=0.3
            )
            
            # Ejecutar
            start_time = time.time()
            coloring, conflict_count = ma.solve(
                max_colors=upper_bound,
                max_generations=min(500, size * 10),
                target_fitness=1.0
            )
            execution_time = time.time() - start_time
            
            # Recopilar resultados
            times.append(execution_time)
            colors.append(len(set(coloring)))
            conflicts.append(conflict_count)
            
            if conflict_count == 0:
                successes += 1
        
        return {
            'avg_time': statistics.mean(times),
            'avg_colors': statistics.mean(colors),
            'avg_conflicts': statistics.mean(conflicts),
            'success_rate': successes / runs
        }
    
    def plot_benchmark_results(self) -> None:
        """Genera visualizaciones de los resultados del benchmark."""
        if not self.results:
            print("❌ No hay resultados para visualizar. Ejecuta benchmark primero.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Resultados del Benchmark - Algoritmo Memético', fontsize=16)
        
        configs = self.results['configurations']
        
        # Gráfico 1: Tiempos de ejecución
        axes[0, 0].bar(range(len(configs)), self.results['execution_times'], 
                      color='skyblue', alpha=0.7)
        axes[0, 0].set_title('⏱️ Tiempos de Ejecución')
        axes[0, 0].set_ylabel('Tiempo (segundos)')
        axes[0, 0].set_xticks(range(len(configs)))
        axes[0, 0].set_xticklabels(configs, rotation=45)
        
        # Gráfico 2: Colores utilizados
        axes[0, 1].bar(range(len(configs)), self.results['colors_used'], 
                      color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('🎨 Colores Utilizados')
        axes[0, 1].set_ylabel('Número de colores')
        axes[0, 1].set_xticks(range(len(configs)))
        axes[0, 1].set_xticklabels(configs, rotation=45)
        
        # Gráfico 3: Conflictos
        axes[1, 0].bar(range(len(configs)), self.results['conflicts'], 
                      color='salmon', alpha=0.7)
        axes[1, 0].set_title('⚡ Conflictos Promedio')
        axes[1, 0].set_ylabel('Número de conflictos')
        axes[1, 0].set_xticks(range(len(configs)))
        axes[1, 0].set_xticklabels(configs, rotation=45)
        
        # Gráfico 4: Tasa de éxito
        success_percentages = [rate * 100 for rate in self.results['success_rate']]
        axes[1, 1].bar(range(len(configs)), success_percentages, 
                      color='gold', alpha=0.7)
        axes[1, 1].set_title('🎯 Tasa de Éxito')
        axes[1, 1].set_ylabel('Porcentaje de éxito (%)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].set_xticks(range(len(configs)))
        axes[1, 1].set_xticklabels(configs, rotation=45)
        
        plt.tight_layout()
        plt.show()


class AlgorithmComparator:
    """Compara el algoritmo memético con otras aproximaciones."""
    
    def __init__(self):
        self.comparison_results = {}
    
    def greedy_coloring(self, graph: Graph) -> Tuple[List[int], int]:
        """
        Implementa algoritmo greedy para coloración de grafos.
        
        Returns:
            Tupla con (coloración, número_de_colores)
        """
        coloring = [-1] * graph.num_vertices
        
        for vertex in range(graph.num_vertices):
            # Encontrar colores usados por vecinos
            used_colors = set()
            for neighbor in graph.get_neighbors(vertex):
                if coloring[neighbor] != -1:
                    used_colors.add(coloring[neighbor])
            
            # Asignar el menor color disponible
            color = 0
            while color in used_colors:
                color += 1
            coloring[vertex] = color
        
        return coloring, max(coloring) + 1 if coloring else 0
    
    def random_coloring(self, graph: Graph, max_colors: int, 
                       max_attempts: int = 1000) -> Tuple[List[int], int]:
        """
        Genera coloraciones aleatorias y selecciona la mejor.
        
        Returns:
            Tupla con (mejor_coloración, conflictos)
        """
        best_coloring = None
        best_conflicts = float('inf')
        
        for _ in range(max_attempts):
            coloring = [np.random.randint(0, max_colors) for _ in range(graph.num_vertices)]
            conflicts = graph.count_conflicts(coloring)
            
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_coloring = coloring.copy()
                
                if conflicts == 0:
                    break
        
        return best_coloring, best_conflicts
    
    def compare_algorithms(self, graph: Graph, max_colors: int = None) -> Dict:
        """
        Compara diferentes algoritmos en el mismo grafo.
        
        Returns:
            Diccionario con resultados de comparación
        """
        if max_colors is None:
            max_colors = graph.get_chromatic_number_upper_bound()
        
        results = {}
        
        print(f"🔍 Comparando algoritmos en grafo de {graph.num_vertices} vértices...")
        
        # 1. Algoritmo Greedy
        print("   🟢 Ejecutando algoritmo greedy...")
        start_time = time.time()
        greedy_coloring, greedy_colors = self.greedy_coloring(graph)
        greedy_time = time.time() - start_time
        greedy_conflicts = graph.count_conflicts(greedy_coloring)
        
        results['greedy'] = {
            'coloring': greedy_coloring,
            'colors_used': greedy_colors,
            'conflicts': greedy_conflicts,
            'time': greedy_time,
            'valid': greedy_conflicts == 0
        }
        
        # 2. Coloración Aleatoria
        print("   🎲 Ejecutando coloración aleatoria...")
        start_time = time.time()
        random_coloring, random_conflicts = self.random_coloring(graph, max_colors)
        random_time = time.time() - start_time
        random_colors = len(set(random_coloring))
        
        results['random'] = {
            'coloring': random_coloring,
            'colors_used': random_colors,
            'conflicts': random_conflicts,
            'time': random_time,
            'valid': random_conflicts == 0
        }
        
        # 3. Algoritmo Memético
        print("   🧬 Ejecutando algoritmo memético...")
        ma = MemeticAlgorithm(
            graph=graph,
            population_size=80,
            mutation_rate=0.1,
            crossover_rate=0.8,
            local_search_prob=0.3
        )
        
        start_time = time.time()
        memetic_coloring, memetic_conflicts = ma.solve(
            max_colors=max_colors,
            max_generations=500,
            target_fitness=1.0
        )
        memetic_time = time.time() - start_time
        memetic_colors = len(set(memetic_coloring))
        
        results['memetic'] = {
            'coloring': memetic_coloring,
            'colors_used': memetic_colors,
            'conflicts': memetic_conflicts,
            'time': memetic_time,
            'valid': memetic_conflicts == 0
        }
        
        self.comparison_results = results
        return results
    
    def print_comparison_table(self) -> None:
        """Imprime tabla comparativa de resultados."""
        if not self.comparison_results:
            print("❌ No hay resultados de comparación disponibles.")
            return
        
        print("\n📊 TABLA COMPARATIVA DE ALGORITMOS")
        print("=" * 70)
        print(f"{'Algoritmo':<15} {'Colores':<8} {'Conflictos':<10} {'Tiempo':<10} {'Válido':<8}")
        print("-" * 70)
        
        algorithms = ['greedy', 'random', 'memetic']
        algorithm_names = ['Greedy', 'Aleatorio', 'Memético']
        
        for alg, name in zip(algorithms, algorithm_names):
            if alg in self.comparison_results:
                result = self.comparison_results[alg]
                valid_icon = "✅" if result['valid'] else "❌"
                print(f"{name:<15} {result['colors_used']:<8} {result['conflicts']:<10} "
                      f"{result['time']:<10.3f} {valid_icon:<8}")
        
        # Determinar ganador
        valid_results = {k: v for k, v in self.comparison_results.items() if v['valid']}
        
        if valid_results:
            best_alg = min(valid_results.items(), 
                          key=lambda x: (x[1]['colors_used'], x[1]['time']))[0]
            print(f"\n🏆 Mejor algoritmo: {best_alg.upper()}")
        else:
            least_conflicts = min(self.comparison_results.items(), 
                                key=lambda x: x[1]['conflicts'])[0]
            print(f"\n🥈 Mejor aproximación: {least_conflicts.upper()}")
    
    def plot_comparison(self) -> None:
        """Genera gráficos comparativos."""
        if not self.comparison_results:
            print("❌ No hay resultados para visualizar.")
            return
        
        algorithms = list(self.comparison_results.keys())
        colors_used = [self.comparison_results[alg]['colors_used'] for alg in algorithms]
        conflicts = [self.comparison_results[alg]['conflicts'] for alg in algorithms]
        times = [self.comparison_results[alg]['time'] for alg in algorithms]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('🔍 Comparación de Algoritmos', fontsize=16)
        
        # Colores utilizados
        bars1 = axes[0].bar(algorithms, colors_used, color=['green', 'orange', 'blue'], alpha=0.7)
        axes[0].set_title('🎨 Colores Utilizados')
        axes[0].set_ylabel('Número de colores')
        for bar, value in zip(bars1, colors_used):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(value), ha='center')
        
        # Conflictos
        bars2 = axes[1].bar(algorithms, conflicts, color=['green', 'orange', 'blue'], alpha=0.7)
        axes[1].set_title('⚡ Conflictos')
        axes[1].set_ylabel('Número de conflictos')
        for bar, value in zip(bars2, conflicts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(value), ha='center')
        
        # Tiempos
        bars3 = axes[2].bar(algorithms, times, color=['green', 'orange', 'blue'], alpha=0.7)
        axes[2].set_title('⏱️ Tiempo de Ejecución')
        axes[2].set_ylabel('Tiempo (segundos)')
        axes[2].set_yscale('log')  # Escala logarítmica para mejor visualización
        for bar, value in zip(bars3, times):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                        f'{value:.3f}', ha='center')
        
        plt.tight_layout()
        plt.show()


def analyze_graph_properties(graph: Graph) -> Dict:
    """
    Analiza propiedades estructurales de un grafo.
    
    Returns:
        Diccionario con propiedades del grafo
    """
    print(f"🔬 Analizando propiedades del grafo...")
    
    # Propiedades básicas
    n_vertices = graph.num_vertices
    n_edges = len(graph.edges)
    max_edges = n_vertices * (n_vertices - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0
    
    # Grados de vértices
    degrees = [len(graph.get_neighbors(v)) for v in range(n_vertices)]
    avg_degree = statistics.mean(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    
    # Estimaciones del número cromático
    greedy_upper_bound = graph.get_chromatic_number_upper_bound()
    clique_lower_bound = max_degree + 1  # Cota inferior simple
    
    properties = {
        'vertices': n_vertices,
        'edges': n_edges,
        'density': density,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'degree_variance': statistics.variance(degrees) if len(degrees) > 1 else 0,
        'greedy_upper_bound': greedy_upper_bound,
        'clique_lower_bound': clique_lower_bound,
        'chromatic_gap': greedy_upper_bound - clique_lower_bound
    }
    
    return properties


def print_graph_analysis(properties: Dict) -> None:
    """Imprime análisis detallado de propiedades del grafo."""
    print("\n🔬 ANÁLISIS ESTRUCTURAL DEL GRAFO")
    print("=" * 45)
    
    print(f"📊 Propiedades Básicas:")
    print(f"   • Vértices: {properties['vertices']}")
    print(f"   • Aristas: {properties['edges']}")
    print(f"   • Densidad: {properties['density']:.3f}")
    
    print(f"\n📈 Análisis de Grados:")
    print(f"   • Grado promedio: {properties['avg_degree']:.2f}")
    print(f"   • Grado máximo: {properties['max_degree']}")
    print(f"   • Grado mínimo: {properties['min_degree']}")
    print(f"   • Varianza de grados: {properties['degree_variance']:.2f}")
    
    print(f"\n🎯 Estimaciones del Número Cromático:")
    print(f"   • Cota inferior (clique): {properties['clique_lower_bound']}")
    print(f"   • Cota superior (greedy): {properties['greedy_upper_bound']}")
    print(f"   • Brecha cromática: {properties['chromatic_gap']}")
    
    # Interpretaciones
    print(f"\n💡 Interpretaciones:")
    if properties['density'] < 0.1:
        print("   • Grafo disperso - algoritmos rápidos esperados")
    elif properties['density'] > 0.7:
        print("   • Grafo denso - mayor dificultad de coloración")
    else:
        print("   • Grafo de densidad media - complejidad moderada")
    
    if properties['chromatic_gap'] <= 2:
        print("   • Brecha cromática pequeña - buena estimación")
    else:
        print("   • Brecha cromática grande - margen de mejora alto")


# Función principal de demostración
def main():
    """Función principal que demuestra las utilidades."""
    print("🧪 SUITE DE ANÁLISIS Y BENCHMARKING")
    print("=" * 50)
    
    # 1. Crear grafo de prueba
    print("\n1️⃣  Creando grafo de prueba...")
    test_graph = create_random_graph(25, 0.4)
    
    # 2. Analizar propiedades
    print("\n2️⃣  Analizando propiedades estructurales...")
    properties = analyze_graph_properties(test_graph)
    print_graph_analysis(properties)
    
    # 3. Comparar algoritmos
    print("\n3️⃣  Comparando algoritmos...")
    comparator = AlgorithmComparator()
    comparison_results = comparator.compare_algorithms(test_graph)
    comparator.print_comparison_table()
    comparator.plot_comparison()
    
    # 4. Benchmark (versión reducida para demostración)
    print("\n4️⃣  Ejecutando benchmark reducido...")
    benchmark = BenchmarkSuite()
    benchmark_results = benchmark.run_benchmark(
        graph_sizes=[10, 15, 20],
        densities=[0.2, 0.4],
        runs_per_config=3
    )
    benchmark.plot_benchmark_results()
    
    print("\n✅ ¡Análisis completo terminado!")


if __name__ == "__main__":
    main()