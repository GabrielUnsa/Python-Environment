import os
import pandas as pd

def recorrer_arbol_completo(directorio, ruta_actual=None, filas=None):
    if ruta_actual is None:
        ruta_actual = []
    if filas is None:
        filas = []

    elementos = sorted(os.listdir(directorio))
    # Guardar la carpeta actual como fila
    filas.append(ruta_actual.copy())

    for elemento in elementos:
        ruta = os.path.join(directorio, elemento)
        if os.path.isdir(ruta):
            recorrer_arbol_completo(ruta, ruta_actual + [elemento], filas)
        else:
            filas.append(ruta_actual + [elemento])
    return filas

if __name__ == "__main__":
    carpeta = input("Ingrese la ruta de la carpeta: ").strip()
    if os.path.exists(carpeta) and os.path.isdir(carpeta):
        # Obtener todas las rutas como listas
        filas = recorrer_arbol_completo(carpeta, [os.path.basename(carpeta)])

        # Determinar la profundidad máxima
        max_niveles = max(len(fila) for fila in filas)

        # Asegurar que todas las filas tengan el mismo número de columnas
        filas = [fila + [""] * (max_niveles - len(fila)) for fila in filas]

        # Crear DataFrame y exportar a Excel
        columnas = [f"Nivel {i+1}" for i in range(max_niveles)]
        df = pd.DataFrame(filas, columns=columnas)
        nombre_excel = "estructura_carpetas.xlsx"
        df.to_excel(nombre_excel, index=False)

        print(f"Estructura guardada en '{nombre_excel}'")
    else:
        print("La ruta ingresada no es válida.")
