import datetime as dt
from enum import Enum
import networkx as nx

# ==========================================
# 1. CORE DATA STRUCTURES
# ==========================================
class LogicType(Enum):
    AND = "AND"
    OR = "OR"

class ConstraintClass(Enum):
    TEMPORAL = 0
    SUCCESS = 1
    GEOMETRY = 2

class Constraint:
    def __init__(self, constraint_class, constraint_type, parent, logic_type=LogicType.AND):
        self.constraint_class = constraint_class
        self.constraint_type = constraint_type
        self.parent = parent
        self.logic_type = logic_type

class ConstrainedObservationRequest:
    def __init__(self, name, task_constraints=[]):
        self.name = name
        self.task_constraints = task_constraints

    def __repr__(self):
        return self.name

class Workflow:
    def __init__(self, requests):
        self.constrained_observation_requests = requests

def build_workflow_graph(workflow):
    graph = nx.MultiDiGraph()
    for req in workflow.constrained_observation_requests:
        graph.add_node(req, name=req.name)
    for req in workflow.constrained_observation_requests:
        for c in req.task_constraints:
            graph.add_edge(c.parent, req, logic_type=c.logic_type)
    return graph


# ==========================================
# 2. SHANNON EXPANSION (DSOP COMPILER)
# ==========================================
class ASTNode:
    """Abstract Syntax Tree for Boolean Logic"""
    def eval(self, env):
        raise NotImplementedError

class VarNode(ASTNode):
    def __init__(self, name):
        self.name = name
    def eval(self, env):
        return env[self.name]

class AndNode(ASTNode):
    def __init__(self, children):
        self.children = children
    def eval(self, env):
        return all(c.eval(env) for c in self.children)

class OrNode(ASTNode):
    def __init__(self, children):
        self.children = children
    def eval(self, env):
        return any(c.eval(env) for c in self.children)

def build_node_success_ast(graph, node):
    """
    Recursively builds the boolean expression for the end-to-end success of a node
    E_node = (Gate_parents) AND S_node
    """
    and_parents = []
    or_parents = []

    for parent in graph.predecessors(node):
        edges = graph.get_edge_data(parent, node)
        for _, edge_data in edges.items():
            logic = edge_data.get('logic_type', LogicType.AND)
            if logic == LogicType.OR:
                or_parents.append(parent)
            else:
                and_parents.append(parent)

    # Local success variable for this node
    local_var = VarNode(f"S_{node.name}")

    if not and_parents and not or_parents:
        return local_var

    clauses = []
    if and_parents:
        clauses.append(AndNode([build_node_success_ast(graph, p) for p in and_parents]))
    if or_parents:
        clauses.append(OrNode([build_node_success_ast(graph, p) for p in or_parents]))

    gate_expr = AndNode(clauses) if len(clauses) > 1 else clauses[0]
    return AndNode([gate_expr, local_var])

def get_ast_vars(ast):
    """Extracts all local variable names from the AST"""
    if isinstance(ast, VarNode):
        return {ast.name}
    elif isinstance(ast, (AndNode, OrNode)):
        res = set()
        for c in ast.children:
            res.update(get_ast_vars(c))
        return res
    return set()

def shannon_expand_dsop(ast, vars_list, current_env={}):
    """
    Recursively performs Shannon Expansion to generate EXACT Disjoint Paths (DSOP).
    f = (v AND f|v=1) OR (NOT v AND f|v=0)
    """
    # Evaluate under current partial assignment
    assigned_vars = set(current_env.keys())
    unassigned = [v for v in vars_list if v not in assigned_vars]

    if not unassigned:
        if ast.eval(current_env) is True:
            return [current_env.copy()]
        return []

    # Pick the next variable to branch on
    var = unassigned[0]

    # Branch 1: Var = True
    env_true = current_env.copy()
    env_true[var] = True
    paths_true = shannon_expand_dsop(ast, vars_list, env_true)

    # Branch 2: Var = False
    env_false = current_env.copy()
    env_false[var] = False
    paths_false = shannon_expand_dsop(ast, vars_list, env_false)

    return paths_true + paths_false


# ==========================================
# 3. MILP EXACT CONSTRAINT GENERATOR
# ==========================================
def generate_exact_milp_gate_constraints(graph, target_node):
    """
    Generates exact linear log-space constraints for Gurobi / OR-Tools
    without any double-counting or overestimation bias.
    """
    # Extract AST for target node's PARENTS gate
    and_parents = []
    or_parents = []
    for p in graph.predecessors(target_node):
        for _, edge_data in graph.get_edge_data(p, target_node).items():
            if edge_data.get('logic_type') == LogicType.OR:
                or_parents.append(p)
            else:
                and_parents.append(p)

    gate_clauses = []
    if and_parents:
        gate_clauses.append(AndNode([build_node_success_ast(graph, p) for p in and_parents]))
    if or_parents:
        gate_clauses.append(OrNode([build_node_success_ast(graph, p) for p in or_parents]))

    gate_ast = AndNode(gate_clauses) if len(gate_clauses) > 1 else gate_clauses[0]

    # Get sorted list of all ancestor variables
    vars_list = sorted(list(get_ast_vars(gate_ast)))

    # Get exact disjoint paths
    disjoint_paths = shannon_expand_dsop(gate_ast, vars_list)

    print(f"\n==================================================")
    print(f" EXACT DSOP COMPILATION FOR: {target_node.name}")
    print(f"==================================================")
    print(f"Ancestor Local Variables: {vars_list}")
    print(f"Number of Disjoint Paths: {len(disjoint_paths)}")

    milp_terms = []
    for idx, path in enumerate(disjoint_paths):
        pos_vars = [v for v, val in path.items() if val is True]
        neg_vars = [v for v, val in path.items() if val is False]

        pos_str = " + ".join([f"ln({v})" for v in pos_vars])
        neg_str = " + ".join([f"ln(1 - {v})" for v in neg_vars])

        log_terms = [t for t in [pos_str, neg_str] if t]
        full_log_expr = " + ".join(log_terms)

        term_var = f"P_{target_node.name}_path_{idx+1}"
        milp_terms.append(term_var)

        print(f"\n  [Path {idx+1} Terms]: {dict(path)}")
        print(f"   └── MILP Log-Space Eq : ln({term_var}) = {full_log_expr}")
        print(f"   └── Real Probability  : {term_var} = exp(ln({term_var}))")

    print(f"\n==================================================")
    print(f" EXACT UNBIASED GATE FORMULA FOR MILP:")
    print(f" A_{target_node.name}^parents = " + " + ".join(milp_terms))
    print(f"==================================================\n")


# ==========================================
# 4. EXECUTION ON THE DIAMOND DAG EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Build Diamond DAG from Figure 1
    t1 = ConstrainedObservationRequest("Task1")
    t2 = ConstrainedObservationRequest("Task2", [Constraint(0, 0, t1, LogicType.AND)])
    t3 = ConstrainedObservationRequest("Task3", [Constraint(0, 0, t1, LogicType.AND)])

    # Task 4 requires Task 1 AND (Task 2 OR Task 3)
    t4 = ConstrainedObservationRequest("Task4", [
        Constraint(0, 0, t1, LogicType.AND),
        Constraint(0, 0, t2, LogicType.OR),
        Constraint(0, 0, t3, LogicType.OR)
    ])

    wf = Workflow([t1, t2, t3, t4])
    graph = build_workflow_graph(wf)

    # Generate exact constraints for Task 4
    generate_exact_milp_gate_constraints(graph, t4)