import networkx as nx
import matplotlib.pyplot as plt
import datetime as dt
from enum import Enum

# =====================================================================
# MINIMAL MOCK-UP OF FAME STRUCTURES TO MAKE THE GRAPH BUILDABLE
# (We don't need real math, just the logical links)
# =====================================================================
class ConstraintClass(Enum):
    TEMPORAL = 1; SUCCESS = 2; GEOMETRY = 3; RESOURCE = 4
class TemporalConstraintType(Enum):
    START_AFTER = 1; START_AFTER_OFFSET = 2; START_BEFORE_OFFSET = 3
class SuccessConstraintType(Enum):
    START_IF_SUCCESSFUL = 1; START_IF_FAILED = 2
class GeometryConstraintType(Enum):
    LLA = 1

class ObservationRequest:
    def __init__(self, name): self.name = name
    def __repr__(self): return self.name

class Constraint:
    def __init__(self, c_class, c_type, parent, params=None):
        self.constraint_class = c_class
        self.constraint_type = c_type
        self.parent = parent; self.params = params

class ConstrainedObservationRequest:
    def __init__(self, name, obs_req, task_constraints=None):
        self.name = name
        self.observation_request = obs_req
        self.task_constraints = task_constraints if task_constraints else []
    def __repr__(self): return self.name

# =====================================================================
# 1. DEFINE YOUR LOGICAL WORKFLOW (Copied logic from your notebook)
# =====================================================================
min_time = dt.datetime.now()
h_3 = dt.timedelta(hours=3); h_6 = dt.timedelta(hours=6)

# ROOT: Initial Scan (RGB)
r_init = ObservationRequest("Rotterdam_Init")
cor_init = ConstrainedObservationRequest("1. Initial RGB", r_init)

# The Immediate Follow-ups (must happen 3-6 hours after INITIAL)
r_sar = ObservationRequest("Rotterdam_SAR_Search")
r_rgb = ObservationRequest("Rotterdam_Heat_Image")

# Follow-up 1 (Branch "A"): Switch to SAR if INITIAL failed (clouds?)
cons_sar_1 = [
    Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET, cor_init, {'offset': h_3}),
    Constraint(ConstraintClass.SUCCESS, SuccessConstraintType.START_IF_FAILED, cor_init),
]
cor_sar_1 = ConstrainedObservationRequest("2a. SAR Search 1", r_sar, cons_sar_1)

# Follow-up 1 (Branch "B"): Zoom with RGB if INITIAL succeeded
cons_rgb_1 = [
    Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET, cor_init, {'offset': h_3}),
    Constraint(ConstraintClass.SUCCESS, SuccessConstraintType.START_IF_SUCCESSFUL, cor_init),
]
cor_rgb_1 = ConstrainedObservationRequest("2b. RGB Image 1", r_rgb, cons_rgb_1)


# The Recursive Follow-ups (must happen after PREVIOUS Search/Image)

# Follow-up 2 (Branch "A"): Search again if Image 1 was successful
# (Target found, it is moving, search where it likely is now)
cons_sar_2 = [
    Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET, cor_rgb_1, {'offset': h_3}),
    Constraint(ConstraintClass.SUCCESS, SuccessConstraintType.START_IF_SUCCESSFUL, cor_rgb_1),
]
cor_sar_2 = ConstrainedObservationRequest("3a. SAR Search 2", r_sar, cons_sar_2)

# Follow-up 2 (Branch "B"): Search again if previous Search 1 failed
cons_sar_3 = [
    Constraint(ConstraintClass.TEMPORAL, TemporalConstraintType.START_AFTER_OFFSET, cor_sar_1, {'offset': h_3}),
    Constraint(ConstraintClass.SUCCESS, SuccessConstraintType.START_IF_FAILED, cor_sar_1),
]
cor_sar_3 = ConstrainedObservationRequest("3b. SAR Search 3", r_sar, cons_sar_3)


# Packaging all constraints together
test_cors = [cor_init, cor_sar_1, cor_rgb_1, cor_sar_2, cor_sar_3]

# =====================================================================
# 2. BUILD THE GRAPH USING FAME logic (Recycled build_workflow_graph logic)
# =====================================================================
G_workflow = nx.DiGraph()

# Define edge colors based on constraint class
class_colors = {
    ConstraintClass.TEMPORAL: 'dimgray',  # Timing constraint
    ConstraintClass.SUCCESS: 'blue',      # 'IF OK' constraint
    ConstraintClass.GEOMETRY: 'lime'      # 'LLA' data constraint
}

for cor in test_cors:
    # Add the target node
    G_workflow.add_node(cor.name)
    
    # Add dependency edges
    for constraint in cor.task_constraints:
        parent_name = constraint.parent.name
        label = f"{constraint.constraint_type.name}"
        if constraint.params:
            if 'offset' in constraint.params:
                label += f" ({constraint.params['offset']})"
        
        G_workflow.add_edge(
            parent_name, cor.name, 
            class_type=constraint.constraint_class,
            label=label,
            color=class_colors[constraint.constraint_class]
        )

# =====================================================================
# 3. DRAW THE GRAPH HIERARCHICALLY
# =====================================================================
plt.figure(figsize=(14, 8))
plt.title("FAME: Operational Dependency Graph for Ship Tracking Workflow", fontsize=16)

# Use NetworkX multipartite layout to create logical hierarchy
G_workflow.nodes["1. Initial RGB"]['layer'] = 0
G_workflow.nodes["2a. SAR Search 1"]['layer'] = 1
G_workflow.nodes["2b. RGB Image 1"]['layer'] = 1
G_workflow.nodes["3a. SAR Search 2"]['layer'] = 2
G_workflow.nodes["3b. SAR Search 3"]['layer'] = 2

pos = nx.multipartite_layout(G_workflow, subset_key="layer")

# Draw nodes with conditional coloring (RGB tasks vs SAR tasks)
node_colors = ['#c7e9b4' if 'RGB' in n else '#fcbba1' for n in G_workflow.nodes()]
nx.draw_networkx_nodes(G_workflow, pos, node_size=3500, node_color=node_colors, edgecolors='black', linewidths=1.5)
nx.draw_networkx_labels(G_workflow, pos, font_size=10, font_weight='bold')

# Draw edges with class coloring
edges = G_workflow.edges(data=True)
colors = [d['color'] for u, v, d in edges]
nx.draw_networkx_edges(G_workflow, pos, edgelist=edges, edge_color=colors, arrowstyle='-|>', arrowsize=25, width=2.0)

# Add detailed edge labels (TIMING OFFSETS)
edge_labels = nx.get_edge_attributes(G_workflow, 'label')
nx.draw_networkx_edge_labels(G_workflow, pos, edge_labels=edge_labels, font_size=8)

# Add a logical legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#c7e9b4', marker='o', label='RGB Task (Imaging)', markersize=10),
    Line2D([0], [0], color='#fcbba1', marker='o', label='SAR Task (Search)', markersize=10),
    Line2D([0], [0], color='dimgray', lw=2, label='Temporal Link (Must start AFTER parent)'),
    Line2D([0], [0], color='blue', lw=2, label='Success Link (Must wait parent status)'),
]
plt.legend(handles=legend_elements, loc='upper right', title="Graph Semantics")

plt.axis('off') # Hide graph axes
plt.tight_layout()
plt.show()