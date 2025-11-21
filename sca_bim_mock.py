import numpy as np
import math
import random

# ==========================================
# MOCK DATA: Simulated BIM Elements (Paper’s Simple Model: 42 elements)
# ==========================================
def generate_mock_elements():
    """Generates 18 columns and 24 beams for testing (matches paper’s simple model)."""
    elements = []
    current_id = 0

    # Ground floor columns (Z: 0–3)
    for i in range(3):
        for j in range(3):
            elements.append({
                'id': current_id, 'type': 'IfcColumn',
                'bbox': {'min_x': i*5, 'max_x': i*5+1, 'min_y': j*5, 'max_y': j*5+1, 'min_z': 0, 'max_z': 3}
            })
            current_id += 1

    # First-floor beams (Z: 3–3.5)
    for i in range(3):  # Horizontal
        elements.append({'id': current_id, 'type': 'IfcBeam',
                         'bbox': {'min_x': 0, 'max_x': 15, 'min_y': i*5, 'max_y': i*5+1, 'min_z': 3, 'max_z': 3.5}})
        current_id += 1
    for j in range(3):  # Vertical
        elements.append({'id': current_id, 'type': 'IfcBeam',
                         'bbox': {'min_x': j*5, 'max_x': j*5+1, 'min_y': 0, 'max_y': 15, 'min_z': 3, 'max_z': 3.5}})
        current_id += 1
    # Add 6 more beams on floor 1
    for _ in range(6):
        elements.append({'id': current_id, 'type': 'IfcBeam',
                         'bbox': {'min_x': random.randint(0, 10), 'max_x': random.randint(11, 15),
                                  'min_y': random.randint(0, 10), 'max_y': random.randint(11, 15),
                                  'min_z': 3, 'max_z': 3.5}})
        current_id += 1

    # First-floor columns (Z: 3.5–6.5)
    for i in range(3):
        for j in range(3):
            elements.append({
                'id': current_id, 'type': 'IfcColumn',
                'bbox': {'min_x': i*5, 'max_x': i*5+1, 'min_y': j*5, 'max_y': j*5+1, 'min_z': 3.5, 'max_z': 6.5}
            })
            current_id += 1

    # Roof beams (Z: 6.5–7)
    for i in range(3):  # Horizontal
        elements.append({'id': current_id, 'type': 'IfcBeam',
                         'bbox': {'min_x': 0, 'max_x': 15, 'min_y': i*5, 'max_y': i*5+1, 'min_z': 6.5, 'max_z': 7}})
        current_id += 1
    for j in range(3):  # Vertical
        elements.append({'id': current_id, 'type': 'IfcBeam',
                         'bbox': {'min_x': j*5, 'max_x': j*5+1, 'min_y': 0, 'max_y': 15, 'min_z': 6.5, 'max_z': 7}})
        current_id += 1
    # Add 6 more roof beams
    for _ in range(6):
        elements.append({'id': current_id, 'type': 'IfcBeam',
                         'bbox': {'min_x': random.randint(0, 10), 'max_x': random.randint(11, 15),
                                  'min_y': random.randint(0, 10), 'max_y': random.randint(11, 15),
                                  'min_z': 6.5, 'max_z': 7}})
        current_id += 1

    return elements

# ==========================================
# 2. MoCC LAYER: Constructability Constraints
# ==========================================
def check_overlap(bbox1, bbox2):
    overlap_x = not (bbox1['max_x'] <= bbox2['min_x'] or bbox1['min_x'] >= bbox2['max_x'])
    overlap_y = not (bbox1['max_y'] <= bbox2['min_y'] or bbox1['min_y'] >= bbox2['max_y'])
    return overlap_x and overlap_y

def generate_mocc(elements):
    n = len(elements)
    mocc = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            a, b = elements[i], elements[j]
            if check_overlap(a['bbox'], b['bbox']):
                if a['bbox']['min_z'] < b['bbox']['min_z']:
                    if a['bbox']['max_z'] >= b['bbox']['min_z'] - 0.5:
                        mocc[i][j] = 1
    return mocc

# ==========================================
# 3. OPTIMIZATION LAYER: Sine Cosine Algorithm (SCA)
# ==========================================
def fitness_function(schedule_vector, mocc):
    total_constraints = np.sum(mocc)
    if total_constraints == 0: return 100.0
    position = {task_id: time for time, task_id in enumerate(schedule_vector)}
    violations = sum(1 for r, c in zip(*np.where(mocc == 1)) if position[r] > position[c])
    return ((total_constraints - violations) / total_constraints) * 100

def run_sca(elements, mocc, population_size=30, max_iter=100):
    n = len(elements)
    population = [np.random.permutation(n) for _ in range(population_size)]
    best_solution, best_fitness = None, -1

    for t in range(max_iter):
        r1 = 2 - (t * 2 / max_iter)
        for i in range(population_size):
            fit = fitness_function(population[i], mocc)
            if fit > best_fitness:
                best_fitness = fit
                best_solution = population[i].copy()
        if best_fitness == 100.0:
            print("Converged to optimal solution!")
            break
        for i in range(population_size):
            r2, r3, r4 = 2 * np.pi * random.random(), 2 * random.random(), random.random()
            X, P = population[i], best_solution
            new_X = X + r1 * (np.sin(r2) if r4 < 0.5 else np.cos(r2)) * np.abs(r3 * P - X)
            population[i] = np.argsort(new_X)
        print(f"Iteration {t}: Best Fitness = {best_fitness:.2f}%")
    return best_solution, best_fitness

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Step 1: Generating Mock BIM Data ---")
    elements = generate_mock_elements()
    print(f"Loaded {len(elements)} elements.")

    print("\n--- Step 2: Generating MoCC ---")
    mocc = generate_mocc(elements)
    print(f"Constraints generated. Total dependencies: {np.sum(mocc)}")

    print("\n--- Step 3: Running SCA Optimization ---")
    best_schedule, score = run_sca(elements, mocc, population_size=50, max_iter=100)

    print("\n--- Final Result ---")
    print(f"Final Constructability Score: {score:.2f}%")
    print("Optimal Construction Sequence (Element IDs):")
    print(best_schedule)

    print("\nSequence Details:")
    for task_id in best_schedule:
        el = next(e for e in elements if e['id'] == task_id)
        print(f"Build {el['type']} (ID: {el['id']}) -> Z: {el['bbox']['min_z']:.1f} to {el['bbox']['max_z']:.1f}")