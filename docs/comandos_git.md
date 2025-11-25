# Guía de Comandos Básicos de Git

Esta guía cubre los comandos esenciales de Git para el flujo de trabajo diario.

## 1. Crear una Rama

### Ver ramas existentes
```bash
git branch
```
Muestra todas las ramas locales. La rama actual aparece con un asterisco (*).

### Crear una nueva rama
```bash
git branch nombre-de-la-rama
```
Crea una nueva rama pero no cambia a ella.

### Crear y cambiar a una nueva rama
```bash
git checkout -b nombre-de-la-rama
```
o con el comando más moderno:
```bash
git switch -c nombre-de-la-rama
```

### Cambiar a una rama existente
```bash
git checkout nombre-de-la-rama
```
o:
```bash
git switch nombre-de-la-rama
```

## 2. Agregar Archivos al Stage

### Ver el estado de los archivos
```bash
git status
```
Muestra qué archivos han sido modificados, agregados o están listos para commit.

### Agregar un archivo específico
```bash
git add nombre-del-archivo.txt
```

### Agregar múltiples archivos
```bash
git add archivo1.txt archivo2.py archivo3.md
```

### Agregar todos los archivos modificados
```bash
git add .
```
Agrega todos los archivos en el directorio actual y subdirectorios.

### Agregar todos los archivos de un tipo específico
```bash
git add *.py
```
Por ejemplo, agrega todos los archivos Python.

### Agregar archivos de forma interactiva
```bash
git add -p
```
Permite revisar y agregar cambios de forma selectiva.

## 3. Hacer Commit

### Commit básico
```bash
git commit -m "Mensaje descriptivo del commit"
```
El mensaje debe ser claro y describir qué cambios se realizaron.

### Commit con mensaje detallado
```bash
git commit
```
Abre el editor de texto configurado para escribir un mensaje más largo.

### Agregar y hacer commit en un solo paso
```bash
git commit -am "Mensaje del commit"
```
Solo funciona con archivos que ya estén siendo rastreados (no archivos nuevos).

### Ver el historial de commits
```bash
git log
```
o para una vista más compacta:
```bash
git log --oneline
```

### Modificar el último commit
```bash
git commit --amend -m "Nuevo mensaje"
```
Útil si olvidaste agregar algo o quieres cambiar el mensaje.

## 4. Push (Enviar Cambios al Repositorio Remoto)

### Ver repositorios remotos
```bash
git remote -v
```
Muestra los repositorios remotos configurados.

### Push de la rama actual
```bash
git push
```
Envía los commits de la rama actual al repositorio remoto.

### Push de una rama nueva (primera vez)
```bash
git push -u origin nombre-de-la-rama
```
o:
```bash
git push --set-upstream origin nombre-de-la-rama
```
El flag `-u` o `--set-upstream` establece la rama remota como predeterminada para futuros push.

### Push a una rama remota específica
```bash
git push origin nombre-de-la-rama
```

### Push forzado (usar con precaución)
```bash
git push --force
```
o más seguro:
```bash
git push --force-with-lease
```
⚠️ Solo usar cuando estés seguro, ya que puede sobrescribir cambios remotos.

## Flujo de Trabajo Típico

```bash
# 1. Crear una nueva rama
git checkout -b feature/nueva-funcionalidad

# 2. Hacer cambios en los archivos...

# 3. Ver qué archivos cambiaron
git status

# 4. Agregar archivos al stage
git add .

# 5. Hacer commit
git commit -m "Agregar nueva funcionalidad"

# 6. Enviar al repositorio remoto
git push -u origin feature/nueva-funcionalidad
```

## Comandos Útiles Adicionales

### Deshacer cambios antes del stage
```bash
git restore nombre-del-archivo
```

### Quitar archivos del stage
```bash
git restore --staged nombre-del-archivo
```

### Ver diferencias
```bash
git diff
```
Muestra los cambios que aún no están en stage.

```bash
git diff --staged
```
Muestra los cambios que están en stage.

### Actualizar rama con cambios remotos
```bash
git pull
```

---

## Buenas Prácticas

1. **Commits frecuentes**: Haz commits pequeños y frecuentes con mensajes descriptivos.
2. **Mensajes claros**: Usa verbos en infinitivo o imperativo: "Agregar", "Corregir", "Actualizar".
3. **Revisar antes de commit**: Usa `git status` y `git diff` para revisar cambios.
4. **Pull antes de Push**: Actualiza tu rama local antes de enviar cambios.
5. **Nombres de rama descriptivos**: Usa nombres como `feature/login`, `fix/bug-123`, `hotfix/security-patch`.
