import ifcopenshell
import ifcopenshell.geom
import numpy as np
import math
import random

# ==========================================
# 1. BIM LAYER: Extract Data from IFC
# ==========================================
def get_bounding_box(shape):
    """Calculates min/max X, Y, Z for a geometry shape."""
    verts = shape.geometry.verts
    # Verts is a flat list [x1, y1, z1, x2, y2, z2...]
    xs = verts[0::3]
    ys = verts[1::3]
    zs = verts[2::3]
    return {
        'min_x': min(xs), 'max_x': max(xs),
        'min_y': min(ys), 'max_y': max(ys),
        'min_z': min(zs), 'max_z': max(zs)
    }

def extract_bim_data(ifc_path):
    """
    Parses IFC file and returns a list of elements with their bounding boxes.
    """
    try:
        model = ifcopenshell.open(ifc_path)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        
        elements = []
        # Focus on Columns and Beams as per the paper
        target_types = ['IfcColumn', 'IfcBeam'] 
        
        idx = 0
        for t in target_types:
            for product in model.by_type(t):
                if product.Representation:
                    try:
                        shape = ifcopenshell.geom.create_shape(settings, product)
                        bbox = get_bounding_box(shape)
                        elements.append({
                            'id': idx,
                            'guid': product.GlobalId,
                            'type': t,
                            'bbox': bbox
                        })
                        idx += 1
                    except:
                        continue
        return elements
    except Exception as e:
        print(f"Error reading IFC (or file not found): {e}")
        print("Generating DUMMY data for testing instead...")
        return generate_dummy_data()

def generate_dummy_data():
    """Generates synthetic columns and beams for testing without an IFC file."""
    elements = []
    # 4 Columns at bottom (Z=0 to 3)
    for i in range(4):
        elements.append({
            'id': i, 'type': 'IfcColumn',
            'bbox': {'min_x': i*5, 'max_x': (i*5)+1, 'min_y': 0, 'max_y': 1, 'min_z': 0, 'max_z': 3}
        })
    # 1 Beam on top of columns (Z=3 to 3.5)
    elements.append({
        'id': 4, 'type': 'IfcBeam',
        'bbox': {'min_x': 0, 'max_x': 20, 'min_y': 0, 'max_y': 1, 'min_z': 3, 'max_z': 3.5}
    })
    return elements

# ==========================================
# 2. MoCC LAYER: Constructability Constraints
# ==========================================
def check_overlap(bbox1, bbox2):
    """Checks if two bounding boxes overlap in XY plane."""
    overlap_x = not (bbox1['max_x'] <= bbox2['min_x'] or bbox1['min_x'] >= bbox2['max_x'])
    overlap_y = not (bbox1['max_y'] <= bbox2['min_y'] or bbox1['min_y'] >= bbox2['max_y'])
    return overlap_x and overlap_y

def generate_mocc(elements):
    """
    Creates the N x N Matrix of Constructability Constraints.
    Logic: If A is below B and they overlap, A must be built before B.
    """
    n = len(elements)
    mocc = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            el_a = elements[i]
            el_b = elements[j]
            
            # Spatial Constraint Logic from Paper:
            # If Element A is strictly below Element B (Z-axis)
            if el_a['bbox']['max_z'] <= el_b['bbox']['min_z']:
                # AND they align vertically (XY overlap)
                if check_overlap(el_a['bbox'], el_b['bbox']):
                    mocc[i][j] = 1 # i must precede j (A -> B)
                    
    return mocc

# ==========================================
# 3. OPTIMIZATION LAYER: Sine Cosine Algorithm (SCA)
# ==========================================
def fitness_function(schedule_vector, mocc):
    """
    Eq. 3 in Paper: Calculate Constructability Score.
    schedule_vector: a permutation of task IDs (order of construction).
    """
    n = len(schedule_vector)
    satisfied_constraints = 0
    total_constraints = np.sum(mocc)
    
    if total_constraints == 0: return 100.0 # No constraints exists
    
    # Map task ID to its position in the schedule
    # position[task_id] = time_slot
    position = {task_id: time for time, task_id in enumerate(schedule_vector)}
    
    violations = 0
    # Check every dependency in MoCC
    rows, cols = np.where(mocc == 1)
    for r, c in zip(rows, cols):
        predecessor = r
        successor = c
        
        # If Predecessor is built AFTER Successor, it's a violation
        if position[predecessor] > position[successor]:
            violations += 1
            
    score = ((total_constraints - violations) / total_constraints) * 100
    return score

def run_sca(elements, mocc, population_size=30, max_iter=100):
    """
    Implements the SCA algorithm to sort tasks.
    """
    n_tasks = len(elements)
    
    # Initialize Population (Random permutations of task sequences)
    population = []
    for _ in range(population_size):
        p = list(range(n_tasks))
        random.shuffle(p)
        population.append(np.array(p))
    
    best_solution = None
    best_fitness = -1
    
    # Main SCA Loop
    for t in range(max_iter):
        # Update r1 (Eq. 2) - decreases linearly
        r1 = 2 - (t * (2 / max_iter))
        
        # Evaluate current population
        current_best_in_gen = None
        current_best_fit_in_gen = -1
        
        for i in range(population_size):
            fit = fitness_function(population[i], mocc)
            
            if fit > best_fitness:
                best_fitness = fit
                best_solution = population[i].copy()
            
            if fit > current_best_fit_in_gen:
                current_best_fit_in_gen = fit
                current_best_in_gen = population[i].copy()

        # Update Positions (Eq. 1)
        # Note: SCA is usually for continuous problems. For discrete scheduling (permutations),
        # we apply the math, then re-rank/sort to get valid integers back.
        
        if best_solution is None: best_solution = population[0] # Safety
        
        for i in range(population_size):
            r2 = 2 * np.pi * random.random()
            r3 = 2 * random.random()
            r4 = random.random()
            
            # Apply SCA Math to the vector
            X = population[i]
            P = best_solution
            
            new_X = np.zeros(n_tasks)
            
            if r4 < 0.5:
                # Sine update
                new_X = X + r1 * math.sin(r2) * np.abs(r3 * P - X)
            else:
                # Cosine update
                new_X = X + r1 * math.cos(r2) * np.abs(r3 * P - X)
            
            # CONVERSION TO DISCRETE SCHEDULE (Permutation)
            # The math gives floats. We sort the indices based on these float values 
            # to get a valid sequence of integers (0 to N-1).
            sorted_indices = np.argsort(new_X)
            population[i] = sorted_indices

        print(f"Iteration {t}: Best Fitness = {best_fitness:.2f}%")
        
        if best_fitness == 100.0:
            print("Converged to optimal solution!")
            break
            
    return best_solution, best_fitness

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data (Uses .ifc in file path)
    print("--- Step 1: Extracting BIM Data ---")
    elements = extract_bim_data("Building-Structural.ifc")
    print(f"Loaded {len(elements)} elements.")
    
    # 2. Generate Matrix
    print("\n--- Step 2: Generating MoCC ---")
    mocc = generate_mocc(elements)
    print(f"Constraints generated. Total dependencies: {np.sum(mocc)}")
    
    # 3. Run SCA
    print("\n--- Step 3: Running SCA Optimization ---")
    best_schedule, score = run_sca(elements, mocc, population_size=20, max_iter=50)
    
    # 4. Output
    print("\n--- Final Result ---")
    print(f"Final Constructability Score: {score}%")
    print("Optimal Construction Sequence (Element IDs):")
    print(best_schedule)
    
    # Validation Display
    print("\nSequence Details:")
    for task_id in best_schedule:
        el = next(e for e in elements if e['id'] == task_id)
        print(f"Build {el['type']} (ID: {el['id']}) -> Z-Level: {el['bbox']['min_z']:.1f} to {el['bbox']['max_z']:.1f}")
        