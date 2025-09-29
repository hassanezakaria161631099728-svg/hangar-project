#%%
import os
import pandas as pd
# import vent function
from shared_functions.vent.vent import vent
# if you also want cell_utils functions:
# from shared_functions import cell_utils
excel_path2 = os.path.join(os.path.dirname(__file__),"excel","eurocode.xlsx")
eurocode = pd.ExcelFile(excel_path2)
# --- Excel file path (relative to project root) ---
excel_path = os.path.join(os.path.dirname(__file__), "excel", "hangar.xlsx")
# Load Excel
hangarf = pd.ExcelFile(excel_path)# Read Excel input
geo = pd.read_excel(hangarf, sheet_name="geography attributes")
wzs = pd.read_excel(eurocode, sheet_name="wind zones")
gcs = pd.read_excel(eurocode, sheet_name="ground categories")
ba = pd.read_excel(hangarf, sheet_name="building attributes")
Lx = ba.iloc[0, 2]
Ly = ba.iloc[0, 3]
direction1='wind1'
direction2='wind2'
# Call custom function (you must implement or convert wind1.m to Python)
T1, T2, T3, T4, T5, Troof1, Twall = vent(ba,Lx,Ly,direction1,geo,wzs,gcs)
T6, T7, T8, T9, T10, Troof2,Twall2 = vent(ba,Lx,Ly,direction2,geo,wzs,gcs)
#%%s = snow(geo, ba, Ly, Lx, Lx, Ly)  # daN/m²
# Save Chapter I tables
#Tables = [T1, T2, T3, T4, T5, Troof1, Twall, T6, T7, T8, T9, T10, Troof2]
#sheetNames = ["T1", "T2", "T3", "T4", "T5", "Troof1", "Twall",
#              "T6", "T7", "T8", "T9", "T10", "Troof2"]
#expxlsx(Tables, os.path.join(excelDir, "chapterI.xlsx"), sheetNames)
#%% ---- Chapter II-1 ----
beamT = os.path.join(excelDir, "tableaudesprofiles.xlsx")
chI = os.path.join(excelDir, "chapterI.xlsx")
from shared_functions.panne import panne
Tpanne, T2, loads, acp, combdel, combV, combM, T8, T9, T10 = panne(b1, b2, hangarf, chI, beamT)
Tables = [Tpanne, T2, loads, acp, combdel, combV, combM, T8, T9, T10]
sheetNames = ["Tpanne", "T2", "loads", "acp", "combdel", "combV", "combM", "T8", "T9", "T10"]
expxlsx(Tables, os.path.join(excelDir, "chapterII-1.xlsx"), sheetNames)
# ---- Chapter II-2 ----
from shared_functions.lisse import lisse
from shared_functions.potelet import potelet
Tlisse, T2, T3, T4 = lisse(hangarf, chI, beamT)
T5, Tpotelet, T6, T7, T8, T9 = potelet(Tlisse, hangarf, chI, beamT)
Tables = [Tlisse, T2, T3, T4, T5, Tpotelet, T6, T7, T8, T9]
sheetNames = ["Tlisse", "T2", "T3", "T4", "T5", "Tpotelet", "T6", "T7", "T8", "T9", "T10"]
expxlsx(Tables, os.path.join(excelDir, "chapterII-2.xlsx"), sheetNames)
# ---- Chapter III-1 ----
from shared_functions.windtruss import windtruss
chII_1 = os.path.join(excelDir, "chapterII-1.xlsx")
T1, T2, Ttrav, Tpot, Tdiag, T6, T7, T8, T9, T10, T11, T12, T13 = windtruss(hangarf, beamT, chI, chII_1)
# If you want to export:
# Tables = [T1, T2, Ttrav, Tpot, Tdiag, T6, T7, T8, T9, T10, T11, T12, T13]
# sheetNames = ["T1","T2","Ttraverse","Tpoteau","Tdiagonale","T6","T7","T8","T9","T10","T11","T12","T13"]
# expxlsx(Tables, os.path.join(excelDir,"chapterIII-1.xlsx"), sheetNames)
# ---- Chapter III-2 ----
from shared_functions.bracedframe import bracedframe
chIII_1 = os.path.join(excelDir, "chapterIII-1.xlsx")
T1, T2, Thea, Tcor, T3, T4, T5 = bracedframe(beamT, chIII_1, hangarf)
Tables = [T1, T2, Thea, Tcor, T3, T4, T5]
sheetNames = ["T1", "T2", "Thea", "Tcor", "T3", "T4", "T5"]
expxlsx(Tables, os.path.join(excelDir, "chapterIII-2.xlsx"), sheetNames)
