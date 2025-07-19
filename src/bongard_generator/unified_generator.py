"""
Unified, Rule-Agnostic Bongard-Logo Data Generator
Generates positive and negative scenes for all rules in ALL_BONGARD_RULES using CP-SAT.
"""
from ortools.sat.python import cp_model
import random
from src.bongard_rules import ALL_BONGARD_RULES
from .drawing import ShapeDrawer

# Constants
GRID_MAX = 4
MIN_OBJECTS = 6
MAX_OBJECTS = 9
NUM_SHAPES = 5
NUM_SIZES = 3
NUM_COLORS = 5
NUM_TEXTURES = 6
TIME_LIMIT_MS = 5000

# Index maps (update as needed for your grammar)
SHAPE_IDX = {'circle':0,'square':1,'triangle':2,'star':3,'pentagon':4}
COLOR_IDX = {'red':0,'blue':1,'green':2,'black':3,'white':4}
SIZE_IDX = {'small':0,'medium':1,'large':2}
TEXTURE_IDX = {'none':0,'striped':1,'dotted':2,'crosshatch':3,'gradient':4,'checker':5}

# Scene variable creation
def create_scene_vars(model, num_objs):
    vars_ = {'x': {}, 'y': {}, 'shape': {}, 'size': {}, 'color': {}, 'texture': {}}
    for i in range(num_objs):
        vars_['x'][i]       = model.NewIntVar(0, GRID_MAX, f'x_{i}')
        vars_['y'][i]       = model.NewIntVar(0, GRID_MAX, f'y_{i}')
        vars_['shape'][i]   = model.NewIntVar(0, NUM_SHAPES-1, f'shape_{i}')
        vars_['size'][i]    = model.NewIntVar(0, NUM_SIZES-1, f'size_{i}')
        vars_['color'][i]   = model.NewIntVar(0, NUM_COLORS-1, f'color_{i}')
        vars_['texture'][i] = model.NewIntVar(0, NUM_TEXTURES-1, f'texture_{i}')
    return vars_

# Scene-level constraints
def add_scene_level_constraints(model, vars_, num_objs):
    for i in range(num_objs):
        for j in range(i+1, num_objs):
            dx = model.NewIntVar(-GRID_MAX, GRID_MAX, f'dx_{i}_{j}')
            dy = model.NewIntVar(-GRID_MAX, GRID_MAX, f'dy_{i}_{j}')
            model.Add(dx == vars_['x'][i] - vars_['x'][j])
            model.Add(dy == vars_['y'][i] - vars_['y'][j])
            model.AddBoolOr([dx > 0, dx < 0, dy > 0, dy < 0])

# AST-to-constraint translator
def add_literal_constraints(model, vars_, literals):
    for lit in literals:
        f, v = lit['feature'], lit['value']
        if f == 'color':
            idx = COLOR_IDX.get(v, 0)
            for i in vars_['color']:
                model.Add(vars_['color'][i] == idx)
        elif f == 'shape':
            idx = SHAPE_IDX.get(v, 0)
            for i in vars_['shape']:
                model.Add(vars_['shape'][i] == idx)
        elif f == 'size':
            idx = SIZE_IDX.get(v, 0)
            for i in vars_['size']:
                model.Add(vars_['size'][i] == idx)
        elif f == 'texture':
            idx = TEXTURE_IDX.get(v, 0)
            for i in vars_['texture']:
                model.Add(vars_['texture'][i] == idx)
        elif f == 'count_op':
            bools = []
            idx = SHAPE_IDX.get(v, 0)
            for i in vars_['shape']:
                b = model.NewBoolVar(f'is_{v}_{i}')
                model.Add(vars_['shape'][i] == idx).OnlyEnforceIf(b)
                model.Add(vars_['shape'][i] != idx).OnlyEnforceIf(b.Not())
                bools.append(b)
            op, cnt = lit['count_op'], lit['count_value']
            if op == 'EQ':
                model.Add(sum(bools) == cnt)
            elif op == 'GT':
                model.Add(sum(bools) > cnt)
            elif op == 'LT':
                model.Add(sum(bools) < cnt)

# Block duplicate solutions
def block_previous_solution(model, vars_, sol):
    clause = []
    for f in ['x','y','shape','size','color','texture']:
        for i, var in vars_[f].items():
            clause.append(var != sol[f][i])
    model.AddBoolOr(clause)

# Main orchestrator
def generate_all_bongard_data(N_per_rule=50):
    dataset = []
    from .drawing import BongardRenderer
    renderer = BongardRenderer(canvas_size=128)
    for rule in ALL_BONGARD_RULES:
        for polarity in ('pos','neg'):
            literals = getattr(rule, 'pos_literals', []) if polarity=='pos' else getattr(rule, 'neg_literals', [])
            attempts = 0
            collected = []
            while len(collected) < N_per_rule and attempts < N_per_rule*3:
                model = cp_model.CpModel()
                num_objs = random.randint(MIN_OBJECTS, MAX_OBJECTS)
                vars_ = create_scene_vars(model, num_objs)
                add_scene_level_constraints(model, vars_, num_objs)
                add_literal_constraints(model, vars_, literals)
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = TIME_LIMIT_MS / 1000
                status = solver.Solve(model)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    sol = {f: {i: solver.Value(var) for i,var in vars_[f].items()} for f in vars_}
                    collected.append(sol)
                    block_previous_solution(model, vars_, sol)
                else:
                    attempts += 1
            # Render and label
            for sol in collected:
                # Convert CP-SAT solution to object dicts for rendering
                objects = []
                for i in range(len(sol['x'])):
                    obj = {
                        'shape': list(SHAPE_IDX.keys())[list(SHAPE_IDX.values()).index(sol['shape'][i])],
                        'position': (sol['x'][i], sol['y'][i]),
                        'size': 30,  # Default size, can be mapped from sol['size'][i]
                        'fill': 'solid',
                        'color': 'black'
                    }
                    objects.append(obj)
                img = renderer.render_scene(objects, canvas_size=128, background_color='white', output_format='numpy')
                dataset.append({
                    'rule': rule.name,
                    'polarity': polarity,
                    'scene': sol,
                    'image': img
                })
    return dataset

# Example usage
if __name__ == "__main__":
    data = generate_all_bongard_data(N_per_rule=10)
    print(f"Generated {len(data)} scenes.")
