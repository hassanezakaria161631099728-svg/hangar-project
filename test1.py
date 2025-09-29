# %% FEM test
import sys, os
import numpy as np
from shared_functions.FEM2D import FEM2D   # make sure FEM2D.py is in the same folder
# --- Define a simple triangular truss (3 nodes, 3 elements) ---
# Node coordinates (x, y) in meters
nodes = np.array([
    [0, 0],   # Node 1
    [4, 0],   # Node 2
    [2, 3]    # Node 3
])
# Connectivity (node1, node2)
elements = np.array([
    [1, 2],  # bottom chord
    [1, 3],  # left diagonal
    [2, 3]   # right diagonal
])
# Cross-sectional areas [m²]
A_h = 0.01   # horizontal members
A_v = 0.01   # vertical members
A_d = 0.01   # diagonals
# Loads: [dof, force]  (dof numbers start at 1 like in MATLAB)
# Node 3 is loaded downward in y-direction (dof = 2*3 = 6)
loads = np.array([
    [6, -100.0]   # 100 kN downward at node 3
])
# Constraints: fix node 1 (dof 1,2) and node 2 in y only (dof 4)
constraints = [1, 2, 4]
# --- Run FEM2D ---
u, reactions, axial_forces = FEM2D(A_h, A_v, A_d, nodes, elements, loads, constraints)
#%%
import pandas as pd
# Define the output folder
output_folder = r"C:\python\excel"
os.makedirs(output_folder, exist_ok=True)
# Build the displacement table as a vertical vector
dof_numbers = np.arange(1, len(u) + 1)  # 1..6
df = pd.DataFrame({
    "DOF": dof_numbers,
    "Displacement": u
})
# Build full file path
file_path = os.path.join(output_folder, "displacements.xlsx")
# Save to Excel
df.to_excel(file_path, index=False, engine="openpyxl")

# %%
from shared_functions import MVD
M,V,D=MVD("uniform", "y",100,6,400)

# %% wind chapter I 
import pandas as pd
import numpy as np
import os
from shared_functions.vent.vent import vent,dimensions
from shared_functions.expxlsx import expxlsx
#from shared_functions.vent.surfaces import wall_parallel,wall_perpendicular,roof_array,roof_list
#from shared_functions.vent.cpe import wall,s_cpe_wall_array,cpe_from_s,roof,s_cpe_roof_array
#from shared_functions.vent.cpi import cpi
#from shared_functions.vent.action_of_set import xyz,reactions,action_of_set
#from shared_functions.expxlsx import expxlsx
excel_path2 = os.path.join(os.path.dirname(__file__),"excel","eurocode.xlsx")
eurocode = pd.ExcelFile(excel_path2)
# --- Excel file path (relative to project root) ---
excel_path = os.path.join(os.path.dirname(__file__), "excel", "hangar.xlsx")
# Load Excel
hangarf = pd.ExcelFile(excel_path)# Read Excel input
geo = pd.read_excel(hangarf,sheet_name="geography attributes")
wzs = pd.read_excel(eurocode,sheet_name="wind zones")
gcs = pd.read_excel(eurocode,sheet_name="ground categories")
ba = pd.read_excel(hangarf,sheet_name="building attributes")
Lx = ba.iloc[0, 3]
Ly = ba.iloc[0, 4]
direction1='wind1'
direction2='wind2'
b,d=dimensions(Lx,Ly,direction1)
bt=ba["bt"].iloc[0]
bt2=ba["bt2"].iloc[0]
T1,T2,T3,T4,T5,Troof,Twall=vent(ba,Lx,Ly,direction1,geo,wzs,gcs)
#T6,T7,T8,T9,T10,Troof2,Twall2=vent(ba,Lx,Ly,direction2,geo,wzs,gcs)
# %% Save Chapter I tables
Tables = [T1,T2,T3,T4,T5,Troof,Twall]
sheetNames = ["T1","T2","T3","T4","T5","Troof","Twall"]
excelDir = r"C:\python\excel"
expxlsx(Tables, os.path.join(excelDir,"chapterI.xlsx"), sheetNames)

# %%
import sys
import numpy as np
# --- Add rootDir (C:\python) to sys.path ---
rootDir = r"C:\python"
if rootDir not in sys.path:
    sys.path.insert(0, rootDir)
# --- Now import the function ---
from shared_functions.draw_beam_bending import draw_beam_bending
# --- Beam definition ---
L = 6.0
h = 0.3
b = 0.25
density = 2500        # concrete
E = 30e9              # concrete ~30 GPa
q_ext_kN_per_m = 7.0  # extra dead load  kN/m
point_loads = [(15.0, 3.0)]  # single 15 kN at midspan
n_points = 600
scale = 1000.0
figsize = (10, 3)
# --- Run beam bending ---
x, deflection = draw_beam_bending(
    L=L, h=h, b=b, density=density, g=9.81, E=E,
    q_ext_kN_per_m=q_ext_kN_per_m, point_loads=point_loads,
    n_points=n_points, scale=scale, draw_deformed_outline=True,
    figsize=figsize
)
# --- Results ---
print("Max deflection (m):", np.max(deflection))
print("Max deflection (mm):", np.max(deflection) * 1000.0)

# %% table test
import  os
import pandas as pd
# create a sample dataframe
T=pd.DataFrame({
    "GroundCat": ["0", "I", "II", "III", "IV"],
    "Col2": [10, 20, 30, 40, 50],
    "Col3": [5, 15, 25, 35, 45],
    "Col4": [100, 200, 300, 400, 500]})

# %%
# test_vent.py
import pandas as pd
import numpy as np
# Import the function
from shared_functions.vent import vent
# ---- dummy shared_functions stubs (if real ones are missing) ----
# You can delete these if you already have implementations
import sys, types
shared = types.ModuleType("shared_functions")
sys.modules["shared_functions"] = shared
sys.modules["shared_functions.cell_utils"] = types.SimpleNamespace(get_cell_like=lambda arr,r,c: 1)

sys.modules["shared_functions.vent.vent_direction"] = types.SimpleNamespace(
    vent_direction=lambda ba,Lx,Ly,direction: (1.0, 2.0)
)
sys.modules["shared_functions.vent.vent_pression_dynamique_de_pointe"] = types.SimpleNamespace(
    vent_pression_dynamique_de_pointe=lambda b,d,geo,ba,bt: (np.array([[0,0,0,1]]), "dummy")
)
sys.modules["shared_functions.vent.vent_murs_perpendiculaires"] = types.SimpleNamespace(
    vent_murs_perpendiculaires=lambda bt,ba,b: (1,2,3,4,5)
)
sys.modules["shared_functions.vent.vent_murs_parallel"] = types.SimpleNamespace(
    vent_murs_parallel=lambda h1,h2,b,d,bt,e: (0.1,0.2,0.3,0.4)
)
sys.modules["shared_functions.vent.vent_cpe_mur"] = types.SimpleNamespace(
    vent_cpe_mur=lambda: [[0.5,0.6]]
)
sys.modules["shared_functions.vent.vent_determination_cpe_selon_surface"] = types.SimpleNamespace(
    vent_determination_cpe_selon_surface=lambda n,Tmur,sm: [0.7,0.8]
)
sys.modules["shared_functions.vent.vent_surfaces_cpe_mur"] = types.SimpleNamespace(
    vent_surfaces_cpe_mur=lambda sm,cpem: ([1],[2],[3])
)
sys.modules["shared_functions.vent.vent_surfaces_zones_toiture_liste"] = types.SimpleNamespace(
    vent_surfaces_zones_toiture_liste=lambda b,d,ba,bt,bt2,e: (1,2,3,4,5)
)
sys.modules["shared_functions.vent.vent_surfaces_zones_toiture_par_cas"] = types.SimpleNamespace(
    vent_surfaces_zones_toiture_par_cas=lambda sf,sg,sh,sJ,sI,bt,bt2,b,ba: (6,7)
)
sys.modules["shared_functions.vent.vent_cpe_toiture"] = types.SimpleNamespace(
    vent_cpe_toiture=lambda ba,bt,bt2,b: [[1,2]]
)
sys.modules["shared_functions.vent.vent_surfaces_cpe_toiture"] = types.SimpleNamespace(
    vent_surfaces_cpe_toiture=lambda Ttoiture,b,ba,st,bt,bt2: ([1],[2],[3])
)
sys.modules["shared_functions.vent.vent_cpi"] = types.SimpleNamespace(
    vent_cpi=lambda bt,bt2,ba,b,d,n: ([0.1],[0.2])
)
sys.modules["shared_functions.vent.vent_pression_aerodynamique_sur_surfaces"] = types.SimpleNamespace(
    vent_pression_aerodynamique_sur_surfaces=lambda cpem,sm,T2: [42]
)
sys.modules["shared_functions.vent.vent_action_ensemble"] = types.SimpleNamespace(
    vent_action_ensemble=lambda *args, **kwargs: ("T4_dummy", "T5_dummy")
)
ba=pd.DataFrame([["a","b","c","d","e","flat roof","dummy_bt2"]])
# ---- run the function ----
if __name__ == "__main__":
    T1, T2, T3, T4,T5,Ttoiture,Tmur=vent(ba=ba,Lx=10.0,Ly=20.0,direction="N",geo={"region": "test"})
#%%
# %% wind chapter I 
import pandas as pd
import numpy as np
import os
from shared_functions.vent.vent import vent
#from shared_functions.expxlsx import expxlsx
excel_path2 = os.path.join(os.path.dirname(__file__),"excel","eurocode.xlsx")
eurocode = pd.ExcelFile(excel_path2)
# --- Excel file path (relative to project root) ---
excel_path = os.path.join(os.path.dirname(__file__), "excel", "hangar.xlsx")
# Load Excel
hangarf = pd.ExcelFile(excel_path)# Read Excel input
geo = pd.read_excel(hangarf,sheet_name="geography attributes")
wzs = pd.read_excel(eurocode,sheet_name="wind zones")
gcs = pd.read_excel(eurocode,sheet_name="ground categories")
ba = pd.read_excel(hangarf,sheet_name="building attributes")
Lx = ba.iloc[0, 3]
Ly = ba.iloc[0, 4]
direction1='wind1'
direction2='wind2'
T1,T2,T3,T4,Troof,Twall=vent(ba,Lx,Ly,direction1,geo,wzs,gcs)

# %%
