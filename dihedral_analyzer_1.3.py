#!/usr/bin/env python3
"""
Dihedral Angle Analysis Script v1.3

This script analyzes dihedral angles and distances from molecular dynamics trajectories.
Supports unlimited custom angles and distances with user-defined names.
Includes functionality to remove periodic boundary artifacts.
"""
import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.transformations import unwrap, center_in_box, wrap
import argparse
import csv
from tqdm import tqdm

def parse_indices(indices_str):
    try:
        indices = np.array([int(x.strip()) for x in indices_str.split(',')])
        if len(indices) != 4:
            raise ValueError(f"Expected 4 indices, got {len(indices)}")
        indices -= 1
        return indices
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid indices format: {e}")

def parse_distance_indices(indices_str):
    try:
        indices = np.array([int(x.strip()) for x in indices_str.split(',')])
        if len(indices) != 2:
            raise ValueError(f"Expected 2 indices, got {len(indices)}")
        indices -= 1
        return indices
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid distance indices format: {e}")

def remove_pbc_artifacts(tpr_file, trr_file):
    """
    Удаляет артефакты элементарной ячейки из траектории.
    """
    print("Загрузка и угадывание связей...")
    u = mda.Universe(tpr_file, trr_file)
    u.atoms.guess_bonds()  # угадываем топологию

    # Показываем пользователю доступные молекулы
    unique_molecules = np.unique(u.atoms.resnames)
    print("\nДоступные молекулы в системе:")
    for i, resname in enumerate(unique_molecules):
        residues = u.select_atoms(f"resname {resname}")
        n_residues = len(np.unique(residues.resids))
        n_atoms = len(residues)
        print(f"{i+1}. {resname}: {n_residues} молекул, {n_atoms} атомов")

    # Запрашиваем выбор молекулы для центрирования
    while True:
        try:
            choice = input(f"\nВыберите номер молекулы для центрирования (1-{len(unique_molecules)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(unique_molecules):
                center_resname = unique_molecules[choice_idx]
                break
            else:
                print(f"Пожалуйста, введите число от 1 до {len(unique_molecules)}")
        except ValueError:
            print("Пожалуйста, введите корректное число")

    print(f"Выбрана молекула: {center_resname}")

    # Селекции
    center_mol = u.select_atoms(f"resname {center_resname}")
    non_water = u.select_atoms("not resname SOL")  # без воды

    print(f"Молекула для центрирования: {len(center_mol)} атомов")
    print(f"Система без воды: {len(non_water)} атомов")

    # Трансформации с unwrap
    transformations = [
        unwrap(u.atoms),                              # разворачиваем все
        center_in_box(center_mol, center='geometry'), # центрируем выбранную молекулу
        wrap(u.atoms)                                 # заворачиваем обратно
    ]

    u.trajectory.add_transformations(*transformations)

    # Создаем временные файлы
    temp_tpr = "temp_centered_no_water.gro"
    temp_trr = "temp_centered_no_water.trr"

    # Сохраняем все кадры без воды в TRR с progress bar
    print("Сохранение траектории без воды...")
    with mda.Writer(temp_trr, n_atoms=len(non_water)) as writer:
        for ts in tqdm(u.trajectory, desc="Обработка кадров", unit="кадр"):
            writer.write(non_water)

    # Сохраняем топологию (только первый кадр)
    non_water.write(temp_tpr)

    print("✅ Обработка PBC завершена!")
    print(f"Временные файлы:")
    print(f"  - {temp_trr} (полная траектория без воды)")
    print(f"  - {temp_tpr} (топология)")

    return temp_tpr, temp_trr

def analyze_dihedrals(tpr_file, trr_file, angle_dict, output_prefix="dihedral_analysis", output_dir=None):
    """
    Анализирует произвольное количество углов с пользовательскими названиями.
    """
    # Создаем полный путь к файлу с учетом output_dir
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        csv_filename = os.path.join(output_dir, f"{output_prefix}_angles.csv")
    else:
        # Создаем директорию для выходных файлов, если она не существует
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        csv_filename = f"{output_prefix}_angles.csv"
    
    print(f"Loading universe from {tpr_file} and {trr_file}")
    u = mda.Universe(tpr_file, trr_file)
    angle_names = []
    angle_arrays = []
    
    # Считаем каждый угол с progress bar
    for name, indices in tqdm(angle_dict.items(), desc="Анализ диэдральных углов", unit="угол"):
        print(f"Analyzing {name} dihedral with indices {indices+1} (1-based)")
        group = u.atoms[indices]
        dih = Dihedral([group]).run()
        angles = dih.results.angles.flatten()
        print(f"Calculated {name} angles")
        angle_names.append(name)
        angle_arrays.append(angles)
    
    # Создаем DataFrame из собранных данных с номерами фреймов
    n_frames = len(angle_arrays[0]) if angle_arrays else 0
    frame_numbers = np.arange(n_frames)
    
    # Сохраняем DataFrame в CSV файл с номерами фреймов
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame"] + angle_names)
        for i in range(n_frames):
            row = [i] + [angle_arrays[j][i] for j in range(len(angle_arrays))]
            writer.writerow(row)
    
    print(f"Analysis results saved to {csv_filename}")

def analyze_distances(tpr_file, trr_file, distance_dict, output_prefix="dihedral_analysis", output_dir=None):
    """
    Анализирует произвольное количество расстояний с пользовательскими названиями.
    """
    # Создаем полный путь к файлу с учетом output_dir
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        csv_filename = os.path.join(output_dir, f"{output_prefix}_distances.csv")
    else:
        # Создаем директорию для выходных файлов, если она не существует
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        csv_filename = f"{output_prefix}_distances.csv"
    
    print(f"Loading universe from {tpr_file} and {trr_file}")
    u = mda.Universe(tpr_file, trr_file)
    distance_names = []
    distance_arrays = []
    
    # Считаем каждое расстояние с progress bar
    for name, indices in tqdm(distance_dict.items(), desc="Анализ расстояний", unit="расстояние"):
        print(f"Calculating {name} distance between atoms {indices+1} (1-based)...")
        atom1 = u.atoms[indices[0]]
        atom2 = u.atoms[indices[1]]
        distances = []
        for ts in u.trajectory:
            d = np.linalg.norm(atom1.position - atom2.position)
            distances.append(d)
        distances = np.array(distances)
        print(f"Calculated {name} distances")
        distance_names.append(name)
        distance_arrays.append(distances)
    
    if not distance_names:
        print("Не задано ни одно расстояние для анализа.")
        return None
    
    # Сохраняем дистанции в CSV
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame"] + distance_names)
        for i in range(len(distance_arrays[0])):
            row = [i] + [distance_arrays[j][i] for j in range(len(distance_arrays))]
            writer.writerow(row)
    print(f"Saved distances to {csv_filename}")
    
    # Статистика
    print("\nSummary Statistics:")
    for name, arr in zip(distance_names, distance_arrays):
        print(f"{name} mean: {np.mean(arr):.3f} Å, std: {np.std(arr):.3f} Å")
    
    return distance_arrays

def main():
    parser = argparse.ArgumentParser(
        description="Analyze dihedral angles and/or distances from MD simulations",
        epilog="""
Examples:
  # Custom named angles and distances:
  dihedral_analyzer_1.3.py topology.tpr trajectory.trr --my_angle "1,2,3,4" --my_angle_another "5,6,7,8" --my_distance "1,2" --my_distance_another "1,3"
  
  # Multiple angles and distances:
  dihedral_analyzer_1.3.py topology.tpr trajectory.trr --angle1 "1,2,3,4" --angle2 "5,6,7,8" --angle3 "9,10,11,12" --dist1 "1,2" --dist2 "3,4" --dist3 "5,6"
  
  # Custom output file names:
  dihedral_analyzer_1.3.py topology.tpr trajectory.trr --angle1 "1,2,3,4" --dist1 "1,2" --angles-output "my_angles" --distances-output "my_distances"
  
  # Specify output directory:
  dihedral_analyzer_1.3.py topology.tpr trajectory.trr --angle1 "1,2,3,4" --dist1 "1,2" --output-dir "results"
  
  # Remove PBC artifacts (use only if artifacts detected, takes 20min-2h):
  dihedral_analyzer_1.3.py topology.tpr trajectory.trr --angle1 "1,2,3,4" --remove-pbc
  
  # Remove PBC artifacts and clean temporary files:
  dihedral_analyzer_1.3.py topology.tpr trajectory.trr --angle1 "1,2,3,4" --remove-pbc --clean-temp
  
  # Use existing temporary files (skip PBC processing):
  dihedral_analyzer_1.3.py temp_centered_no_water.gro temp_centered_no_water.trr --angle1 "1,2,3,4"
        """
    )
    parser.add_argument("tpr_file", help="Path to topology file (.tpr or .gro)")
    parser.add_argument("trr_file", help="Path to trajectory file (.trr)")
    parser.add_argument("--output", "-o", default="dihedral_analysis",
                        help="Prefix for output files (default: dihedral_analysis)")
    parser.add_argument("--angles-output", default=None,
                        help="Prefix for angles output file (overrides --output)")
    parser.add_argument("--distances-output", default=None,
                        help="Prefix for distances output file (overrides --output)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output files (default: current directory)")
    parser.add_argument("--remove-pbc", action="store_true",
                        help="Remove periodic boundary artifacts. Use only if artifacts detected. Takes 20min-2h depending on system size.")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files after PBC processing (default: files are kept)")
    parser.add_argument("--clean-temp", action="store_true",
                        help="Remove temporary files after PBC processing")
    
    # Пользовательские углы и расстояния будут добавляться динамически
    args, unknown = parser.parse_known_args()
    
    # Обрабатываем неизвестные аргументы для пользовательских углов и расстояний
    custom_angles = {}
    custom_distances = {}
    
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith('--'):
            name = arg[2:]  # убираем --
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                # Это пользовательский угол (4 индекса)
                try:
                    indices = parse_indices(unknown[i + 1])
                    custom_angles[name] = indices
                    print(f"Added custom angle '{name}' with indices {indices+1}")
                    i += 2
                except Exception as e:
                    # Возможно, это расстояние (2 индекса)
                    try:
                        indices = parse_distance_indices(unknown[i + 1])
                        custom_distances[name] = indices
                        print(f"Added custom distance '{name}' with indices {indices+1}")
                        i += 2
                    except Exception as e2:
                        print(f"Warning: Could not parse argument {arg} {unknown[i + 1]}: {e}, {e2}")
                        i += 2
            else:
                i += 1
        else:
            i += 1
    
    # Проверяем существование входных файлов
    if not os.path.exists(args.tpr_file):
        print(f"Ошибка: файл топологии '{args.tpr_file}' не найден.")
        return 1
    if not os.path.exists(args.trr_file):
        print(f"Ошибка: файл траектории '{args.trr_file}' не найден.")
        return 1
    
    # Обработка PBC артефактов если запрошено
    tpr_file = args.tpr_file
    trr_file = args.trr_file
    
    # Проверяем наличие существующих временных файлов
    temp_files_exist = (os.path.exists("temp_centered_no_water.gro") and 
                       os.path.exists("temp_centered_no_water.trr"))
    
    if temp_files_exist and not args.remove_pbc:
        print("📁 Обнаружены существующие временные файлы PBC обработки:")
        print("  - temp_centered_no_water.gro")
        print("  - temp_centered_no_water.trr")
        print("💡 Используйте эти файлы для анализа без повторной PBC обработки.")
        print("   Или используйте --remove-pbc для создания новых файлов.")
    
    if args.remove_pbc:
        print("⚠️  ВНИМАНИЕ: Удаление PBC артефактов может занять от 20 минут до 2 часов!")
        print("Этот процесс создаст временные файлы в текущей директории.")
        proceed = input("Продолжить? (y/N): ")
        if proceed.lower() != 'y':
            print("Операция отменена.")
            return 0
        
        try:
            tpr_file, trr_file = remove_pbc_artifacts(args.tpr_file, args.trr_file)
        except Exception as e:
            print(f"Ошибка при удалении PBC артефактов: {e}")
            return 1
    
    # Определяем префиксы для файлов
    angles_prefix = args.angles_output if args.angles_output is not None else args.output
    distances_prefix = args.distances_output if args.distances_output is not None else args.output
    
    try:
        if custom_angles:
            analyze_dihedrals(tpr_file, trr_file, custom_angles, angles_prefix, args.output_dir)
        
        if custom_distances:
            analyze_distances(tpr_file, trr_file, custom_distances, distances_prefix, args.output_dir)
        
        if not custom_angles and not custom_distances:
            print("Ошибка: необходимо указать хотя бы один угол или расстояние.")
            print("Примеры:")
            print("  --my_angle 1,2,3,4 --my_distance 1,2")
            print("  --angle1 1,2,3,4 --angle2 5,6,7,8 --dist1 1,2 --dist2 3,4")
            return 1
    except Exception as e:
        print(f"Ошибка при выполнении анализа: {e}")
        return 1
    
    # Обработка временных файлов
    if args.remove_pbc:
        if args.clean_temp:
            # Удаляем временные файлы
            try:
                if os.path.exists("temp_centered_no_water.trr"):
                    os.remove("temp_centered_no_water.trr")
                if os.path.exists("temp_centered_no_water.gro"):
                    os.remove("temp_centered_no_water.gro")
                print("✅ Временные файлы удалены.")
            except Exception as e:
                print(f"⚠️  Предупреждение: не удалось удалить временные файлы: {e}")
        else:
            # Сохраняем временные файлы (по умолчанию)
            print("\n📁 Временные файлы сохранены:")
            if os.path.exists("temp_centered_no_water.trr"):
                size_trr = os.path.getsize("temp_centered_no_water.trr") / (1024*1024)  # MB
                print(f"  - temp_centered_no_water.trr ({size_trr:.1f} MB)")
            if os.path.exists("temp_centered_no_water.gro"):
                size_gro = os.path.getsize("temp_centered_no_water.gro") / 1024  # KB
                print(f"  - temp_centered_no_water.gro ({size_gro:.1f} KB)")
            print("\n💡 Эти файлы можно использовать для повторного анализа без PBC обработки.")
            print("   Для удаления выполните: rm temp_centered_no_water.*")
            print("   Или используйте флаг --clean-temp при следующем запуске.")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 