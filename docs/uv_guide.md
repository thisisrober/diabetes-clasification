# Guía de uv - Gestor de Paquetes Python Ultrarrápido

## ¿Qué es uv?

**uv** es un gestor de paquetes y entornos virtuales para Python desarrollado por [Astral](https://astral.sh/) (los creadores de Ruff). Está escrito en Rust y es **10-100x más rápido** que pip y pip-tools.

### Características principales:
- ⚡ **Ultrarrápido**: Instalaciones casi instantáneas gracias a caché agresivo
- 🔒 **Lockfiles**: Genera `uv.lock` para reproducibilidad exacta
- 📦 **Todo en uno**: Reemplaza pip, pip-tools, virtualenv, pyenv y más
- 🐍 **Gestión de Python**: Puede instalar y gestionar versiones de Python
- 🔄 **Compatible**: Funciona con `requirements.txt` y `pyproject.toml`

---

## Archivos Generados por uv

Cuando inicializas un proyecto con uv, se crean los siguientes archivos:

| Archivo | ¿Qué es? | ¿Va al repo? |
|---------|----------|--------------|
| `pyproject.toml` | Configuración del proyecto y dependencias | ✅ **SÍ** |
| `uv.lock` | Lockfile con versiones exactas de todas las dependencias | ✅ **SÍ** |
| `.python-version` | Versión de Python del proyecto | ✅ **SÍ** |
| `.venv/` | Entorno virtual (carpeta pesada) | ❌ **NO** |
| `main.py` | Script de ejemplo (puedes eliminarlo) | Opcional |

### ¿Por qué incluir `uv.lock` en el repo?

El archivo `uv.lock` garantiza que **todos los colaboradores instalen exactamente las mismas versiones** de los paquetes. Esto evita el clásico "en mi máquina funciona".

```
# uv.lock contiene:
# - Versiones exactas de cada paquete
# - Hashes de los archivos descargados
# - Resolución completa del árbol de dependencias
```

---

## Comandos Esenciales de uv

### Instalación de uv

```powershell
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Inicializar un proyecto

```bash
# Crear proyecto nuevo
uv init --name mi-proyecto

# En un directorio existente con requirements.txt
uv init --name mi-proyecto
uv add -r requirements.txt
```

### Gestión de dependencias

```bash
# Agregar paquetes
uv add pandas numpy matplotlib

# Agregar paquete de desarrollo (solo para desarrollo)
uv add --dev pytest black

# Eliminar paquete
uv remove pandas

# Actualizar todos los paquetes
uv lock --upgrade

# Actualizar un paquete específico
uv lock --upgrade-package pandas
```

### Sincronizar entorno

```bash
# Instalar dependencias del lockfile (para colaboradores)
uv sync

# Sincronizar incluyendo dependencias de desarrollo
uv sync --dev
```

### Ejecutar scripts

```bash
# Ejecutar un script Python
uv run python mi_script.py

# Ejecutar Jupyter
uv run jupyter notebook
```

### Gestión de Python

```bash
# Ver versiones de Python disponibles
uv python list

# Instalar una versión específica
uv python install 3.12

# Usar una versión específica en el proyecto
uv python pin 3.12
```

---

## Flujo de Trabajo para Colaboradores

### Cuando clonas el repositorio por primera vez:

```bash
# 1. Clonar el repo
git clone https://github.com/thisisrober/diabetes-clasification.git
cd diabetes-clasification

# 2. Instalar uv (si no lo tienes)
# Windows:
irm https://astral.sh/uv/install.ps1 | iex
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Sincronizar dependencias (crea .venv automáticamente)
uv sync

# 4. ¡Listo! Ya puedes trabajar
uv run jupyter notebook
```

### Cuando agregas una nueva dependencia:

```bash
# 1. Agregar el paquete
uv add nuevo-paquete

# 2. Commitear los cambios
git add pyproject.toml uv.lock
git commit -m "Add nuevo-paquete dependency"
git push
```

### Cuando otro colaborador agregó dependencias:

```bash
# 1. Actualizar el repo
git pull

# 2. Sincronizar dependencias
uv sync
```

---

## Comparación: uv vs pip vs conda

| Característica | uv | pip | conda |
|---------------|-----|-----|-------|
| Velocidad | ⚡⚡⚡ | ⚡ | ⚡ |
| Lockfile nativo | ✅ | ❌ | ❌ |
| Resolución de dependencias | Excelente | Básica | Buena |
| Gestión de Python | ✅ | ❌ | ✅ |
| Entornos virtuales | ✅ | Necesita venv | ✅ |
| Tamaño | ~10MB | ~10MB | ~400MB+ |

---

## Estructura del Proyecto con uv

```
proyecto/
├── .git/
├── .gitignore
├── .python-version      # ✅ Versión de Python (va al repo)
├── .venv/               # ❌ Entorno virtual (NO va al repo)
├── pyproject.toml       # ✅ Configuración y dependencias (va al repo)
├── uv.lock              # ✅ Lockfile (va al repo)
├── requirements.txt     # Opcional, para compatibilidad
├── src/
│   └── ...
└── docs/
    └── ...
```

---

## Tips y Buenas Prácticas

1. **Siempre commitea `uv.lock`**: Garantiza reproducibilidad
2. **Usa `uv sync` en CI/CD**: Es más rápido que `pip install`
3. **Separa dependencias de desarrollo**: Usa `uv add --dev` para pytest, black, etc.
4. **No commitas `.venv/`**: Es pesado y se puede regenerar con `uv sync`

---

## Enlaces Útiles

- 📖 [Documentación oficial de uv](https://docs.astral.sh/uv/)
- 🐙 [Repositorio en GitHub](https://github.com/astral-sh/uv)
- 📦 [Migración desde pip](https://docs.astral.sh/uv/guides/integration/pip/)

---

*Documentación creada para el proyecto Diabetes Classification*
