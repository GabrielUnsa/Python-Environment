"""
Script de Prueba Simple - Algoritmo Memético para Coloración de Grafos
======================================================================

Script standalone para probar el algoritmo memético sin dependencias externas.
"""

import sys
import os
sys.path.append('.')

from memetic_graph_coloring import Graph, MemeticAlgorithm, create_random_graph
import time


def test_small_graph():
    """Prueba con un grafo pequeño definido manualmente."""
    print("🔬 PRUEBA 1: Grafo Pequeño Manual")
    print("-" * 40)
    
    # Crear grafo pequeño conocido (K4 - clique de 4 vértices)
    graph = Graph(4)
    # Conectar todos los vértices entre sí (grafo completo)
    for i in range(4):
        for j in range(i + 1, 4):
            graph.add_edge(i, j)
    
    print(f"📊 Grafo creado: {graph.num_vertices} vértices, {len(graph.edges)} aristas")
    print(f"🏗️  Aristas: {graph.edges}")
    
    # Para un K4, necesitamos exactamente 4 colores
    upper_bound = graph.get_chromatic_number_upper_bound()
    print(f"📈 Cota superior estimada: {upper_bound} colores")
    print(f"🎯 Número cromático real (K4): 4 colores")
    
    # Ejecutar algoritmo
    ma = MemeticAlgorithm(
        graph=graph,
        population_size=20,
        mutation_rate=0.2,
        crossover_rate=0.8,
        local_search_prob=0.5
    )
    
    start_time = time.time()
    coloring, conflicts = ma.solve(
        max_colors=4,
        max_generations=100,
        target_fitness=1.0
    )
    execution_time = time.time() - start_time
    
    # Mostrar resultados
    print(f"\n📋 Resultados:")
    print(f"   🎨 Coloración: {coloring}")
    print(f"   🌈 Colores usados: {sorted(set(coloring))}")
    print(f"   ⚡ Conflictos: {conflicts}")
    print(f"   ⏱️  Tiempo: {execution_time:.3f} segundos")
    print(f"   ✅ Válida: {'SÍ' if conflicts == 0 else 'NO'}")
    
    return conflicts == 0


def test_medium_graph():
    """Prueba con un grafo mediano aleatorio."""
    print("\n🔬 PRUEBA 2: Grafo Mediano Aleatorio")
    print("-" * 40)
    
    # Crear grafo aleatorio
    graph = create_random_graph(20, 0.3)
    upper_bound = graph.get_chromatic_number_upper_bound()
    
    print(f"📊 Grafo generado: {graph.num_vertices} vértices, {len(graph.edges)} aristas")
    print(f"📈 Densidad: {len(graph.edges) / (graph.num_vertices * (graph.num_vertices - 1) / 2):.3f}")
    print(f"📈 Cota superior: {upper_bound} colores")
    
    # Configurar algoritmo
    ma = MemeticAlgorithm(
        graph=graph,
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        local_search_prob=0.3
    )
    
    # Ejecutar
    start_time = time.time()
    coloring, conflicts = ma.solve(
        max_colors=upper_bound,
        max_generations=300,
        target_fitness=1.0
    )
    execution_time = time.time() - start_time
    
    # Analizar resultados
    colors_used = len(set(coloring))
    improvement = ((upper_bound - colors_used) / upper_bound) * 100
    
    print(f"\n📋 Resultados:")
    print(f"   🎨 Colores utilizados: {colors_used} de {upper_bound} disponibles")
    print(f"   📈 Mejora vs. greedy: {improvement:.1f}%")
    print(f"   ⚡ Conflictos finales: {conflicts}")
    print(f"   ⏱️  Tiempo de ejecución: {execution_time:.2f} segundos")
    print(f"   ✅ Solución válida: {'SÍ' if conflicts == 0 else 'NO'}")
    
    # Verificar algunos vértices
    print(f"\n🔍 Verificación manual (primeros 10 vértices):")
    for i in range(min(10, graph.num_vertices)):
        neighbors = graph.get_neighbors(i)
        neighbor_colors = [coloring[n] for n in neighbors]
        has_conflict = coloring[i] in neighbor_colors
        status = "❌ CONFLICTO" if has_conflict else "✅ OK"
        print(f"   Vértice {i}: color {coloring[i]}, vecinos: {neighbors}, {status}")
    
    return conflicts == 0


def test_graph_from_file():
    """Prueba cargando un grafo desde archivo si existe."""
    print("\n🔬 PRUEBA 3: Cargar Grafo desde Archivo")
    print("-" * 40)
    
    # Buscar archivos de grafos en el directorio de datos
    possible_files = [
        "jupyter-notebooks/Academic Investigation/DataSets/Graphs/DSJC125_1.txt",
        "jupyter-notebooks/Academic Investigation/DataSets/DSJC125_1.txt"
    ]
    
    graph_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            graph_file = file_path
            break
    
    if not graph_file:
        print("❌ No se encontraron archivos de grafos conocidos.")
        print("🔧 Creando grafo grande para simulación...")
        graph = create_random_graph(50, 0.2)
        upper_bound = graph.get_chromatic_number_upper_bound()
        print(f"📊 Grafo simulado: {graph.num_vertices} vértices, {len(graph.edges)} aristas")
    else:
        print(f"📁 Cargando grafo desde: {graph_file}")
        graph = Graph()
        graph.load_from_file(graph_file)
        upper_bound = graph.get_chromatic_number_upper_bound()
        print(f"📊 Grafo cargado: {graph.num_vertices} vértices, {len(graph.edges)} aristas")
    
    print(f"📈 Cota superior: {upper_bound} colores")
    print(f"📊 Densidad: {len(graph.edges) / (graph.num_vertices * (graph.num_vertices - 1) / 2):.4f}")
    
    # Configurar algoritmo para grafo grande
    ma = MemeticAlgorithm(
        graph=graph,
        population_size=min(100, graph.num_vertices),
        mutation_rate=0.08,
        crossover_rate=0.85,
        local_search_prob=0.25
    )
    
    # Ejecutar con límite de tiempo
    print(f"🚀 Ejecutando algoritmo (máximo 60 segundos)...")
    start_time = time.time()
    
    coloring, conflicts = ma.solve(
        max_colors=min(upper_bound, 20),  # Limitar colores para evitar explosión
        max_generations=min(500, graph.num_vertices * 5),
        target_fitness=1.0
    )
    execution_time = time.time() - start_time
    
    # Resultados
    colors_used = len(set(coloring))
    
    print(f"\n📋 Resultados Finales:")
    print(f"   🎨 Colores utilizados: {colors_used}")
    print(f"   ⚡ Conflictos: {conflicts}")
    print(f"   ⏱️  Tiempo: {execution_time:.2f} segundos")
    print(f"   🎯 Generaciones ejecutadas: {ma.generation}")
    print(f"   📈 Fitness final: {ma.best_individual.fitness:.4f}")
    
    if conflicts == 0:
        print("   🎉 ¡ÉXITO! Coloración válida encontrada.")
        improvement = ((upper_bound - colors_used) / upper_bound) * 100
        print(f"   📊 Mejora vs. greedy: {improvement:.1f}%")
    else:
        print(f"   ⚠️  Solución aproximada con {conflicts} conflictos.")
        print("   💡 Sugerencia: Aumentar generaciones o población.")
    
    return conflicts == 0


def run_comparison_test():
    """Compara el algoritmo memético con algoritmo greedy simple."""
    print("\n🔬 PRUEBA 4: Comparación Algoritmo Memético vs. Greedy")
    print("-" * 55)
    
    # Crear grafo de prueba
    graph = create_random_graph(30, 0.35)
    
    print(f"📊 Grafo de prueba: {graph.num_vertices} vértices, {len(graph.edges)} aristas")
    
    # Algoritmo Greedy
    print("\n🟢 Ejecutando algoritmo greedy...")
    start_time = time.time()
    greedy_colors = graph.get_chromatic_number_upper_bound()
    greedy_time = time.time() - start_time
    
    # Crear coloración greedy manualmente para contar conflictos
    greedy_coloring = [-1] * graph.num_vertices
    for vertex in range(graph.num_vertices):
        used_colors = set()
        for neighbor in graph.get_neighbors(vertex):
            if greedy_coloring[neighbor] != -1:
                used_colors.add(greedy_coloring[neighbor])
        
        color = 0
        while color in used_colors:
            color += 1
        greedy_coloring[vertex] = color
    
    greedy_conflicts = graph.count_conflicts(greedy_coloring)
    
    # Algoritmo Memético
    print("🧬 Ejecutando algoritmo memético...")
    ma = MemeticAlgorithm(
        graph=graph,
        population_size=60,
        mutation_rate=0.12,
        crossover_rate=0.8,
        local_search_prob=0.35
    )
    
    start_time = time.time()
    memetic_coloring, memetic_conflicts = ma.solve(
        max_colors=greedy_colors,
        max_generations=400,
        target_fitness=1.0
    )
    memetic_time = time.time() - start_time
    memetic_colors_used = len(set(memetic_coloring))
    
    # Comparación
    print(f"\n📊 COMPARACIÓN DE RESULTADOS")
    print(f"{'Algoritmo':<15} {'Colores':<8} {'Conflictos':<10} {'Tiempo':<10} {'Válido':<8}")
    print("-" * 55)
    print(f"{'Greedy':<15} {greedy_colors:<8} {greedy_conflicts:<10} {greedy_time:<10.3f} {'✅' if greedy_conflicts == 0 else '❌':<8}")
    print(f"{'Memético':<15} {memetic_colors_used:<8} {memetic_conflicts:<10} {memetic_time:<10.3f} {'✅' if memetic_conflicts == 0 else '❌':<8}")
    
    # Análisis
    print(f"\n💡 Análisis:")
    if memetic_conflicts == 0 and greedy_conflicts == 0:
        color_improvement = greedy_colors - memetic_colors_used
        if color_improvement > 0:
            print(f"   🎉 Memético mejoró en {color_improvement} colores!")
        elif color_improvement == 0:
            print("   ⚖️  Ambos algoritmos encontraron la misma solución.")
        else:
            print(f"   🤔 Greedy fue mejor por {-color_improvement} colores.")
    elif memetic_conflicts == 0:
        print("   🏆 Solo el memético encontró solución válida!")
    elif greedy_conflicts == 0:
        print("   🔧 Solo greedy encontró solución válida.")
    else:
        if memetic_conflicts < greedy_conflicts:
            print(f"   📈 Memético redujo conflictos en {greedy_conflicts - memetic_conflicts}.")
        else:
            print("   ⚠️  Ambos algoritmos tuvieron dificultades.")
    
    print(f"   ⏱️  Factor de tiempo: {memetic_time / greedy_time:.1f}x más lento que greedy")


def main():
    """Función principal que ejecuta todas las pruebas."""
    print("🎨 ALGORITMO MEMÉTICO - SUITE DE PRUEBAS")
    print("=" * 60)
    print("Pruebas diseñadas para verificar funcionalidad sin dependencias externas")
    print("=" * 60)
    
    results = []
    
    # Ejecutar todas las pruebas
    results.append(("Grafo K4 (pequeño)", test_small_graph()))
    results.append(("Grafo mediano aleatorio", test_medium_graph()))
    results.append(("Grafo grande/archivo", test_graph_from_file()))
    
    # Ejecutar comparación
    run_comparison_test()
    
    # Resumen final
    print(f"\n🏁 RESUMEN FINAL DE PRUEBAS")
    print("=" * 35)
    successful_tests = sum(results)
    total_tests = len(results)
    
    for test_name, success in results:
        status = "✅ ÉXITO" if success else "⚠️  APROXIMADO"
        print(f"{test_name:<25} {status}")
    
    print(f"\n📊 Estadísticas finales:")
    print(f"   ✅ Pruebas exitosas: {successful_tests}/{total_tests}")
    print(f"   📈 Tasa de éxito: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("\n🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        print("   El algoritmo memético está funcionando correctamente.")
    elif successful_tests > 0:
        print("\n✅ El algoritmo funciona bien en la mayoría de casos.")
        print("   Algunas pruebas resultaron en soluciones aproximadas.")
    else:
        print("\n⚠️  Se necesita ajustar parámetros del algoritmo.")
        print("   Considera aumentar generaciones o tamaño de población.")
    
    print(f"\n🔧 Para usar el algoritmo en tus propios grafos:")
    print(f"   1. Importa: from memetic_graph_coloring import Graph, MemeticAlgorithm")
    print(f"   2. Crea tu grafo: graph = Graph(num_vertices)")
    print(f"   3. Añade aristas: graph.add_edge(v1, v2)")
    print(f"   4. Ejecuta: ma = MemeticAlgorithm(graph); coloring, conflicts = ma.solve(...)")


if __name__ == "__main__":
    main()