# 🔧 Исправление ошибки MDAnalysis в PyInstaller

## ❌ Проблема

При запуске скомпилированного exe файла возникает ошибка:
```
ModuleNotFoundError: No module named 'MDAnalysis.lib.formats.cython_util'
```

## ✅ Решение

### Способ 1: Использование готового spec файла (Рекомендуется)

1. **Скопируйте spec файл** `dihedral_analyzer_1.3_windows.spec` в директорию с проектом
2. **Запустите сборку:**
   ```bash
   pyinstaller dihedral_analyzer_1.3_windows.spec
   ```

### Способ 2: Локальная сборка на Windows

1. **Скачайте все файлы проекта**
2. **Запустите batch скрипт:**
   ```cmd
   build_local.bat
   ```

### Способ 3: GitHub Actions (Автоматически)

1. **Перейдите в репозиторий:** https://github.com/pantyushenko/dihedral-analyzer-1754176741
2. **Actions → Build Windows Executable → Run workflow**
3. **Скачайте готовый exe файл**

## 🔍 Что было исправлено

### В spec файле добавлены:
- ✅ Все необходимые MDAnalysis модули
- ✅ Cython утилиты
- ✅ Трансформации
- ✅ Форматы файлов
- ✅ Дополнительные зависимости

### В GitHub Actions:
- ✅ Использование spec файла вместо автоматической генерации
- ✅ Тестирование exe файла после сборки
- ✅ Проверка работоспособности

## 📋 Проверка работоспособности

После сборки протестируйте exe файл:
```cmd
dihedral_analyzer_1.3_windows.exe --help
```

Должна появиться справка без ошибок.

## 🚨 Если проблема остается

### Альтернативные решения:

1. **Использовать Python скрипт напрямую:**
   ```bash
   python dihedral_analyzer_1.3.py --help
   ```

2. **Создать виртуальное окружение:**
   ```bash
   conda create -n mda_env python=3.9
   conda activate mda_env
   pip install MDAnalysis numpy scipy matplotlib tqdm
   python dihedral_analyzer_1.3.py --help
   ```

3. **Использовать Docker:**
   ```bash
   docker run -it --rm -v $(pwd):/work python:3.9 bash
   pip install MDAnalysis numpy scipy matplotlib tqdm
   python dihedral_analyzer_1.3.py --help
   ```

## 📞 Поддержка

Если проблема не решается:
1. Проверьте версию Python (рекомендуется 3.9)
2. Убедитесь в установке всех зависимостей
3. Попробуйте пересобрать spec файл
4. Обратитесь к логам PyInstaller для диагностики

---

**🎯 Результат: Исправленная сборка с правильными зависимостями MDAnalysis!** 