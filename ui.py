# ---------- codebook_mapping.py ----------
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.io as pio

from src import *
from ui_sliders import LAB_VARIABLES, build_lab_sliders


# === Codebook: label + choices (value codes match NHANES) ===
# Notes:
# - Most disease items are 1=Yes, 2=No (default to No).
# - DIQ010 includes 3=Borderline (counts as Yes in fs1).
# - HUQ010: 1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor (fs2 uses Fair/Poor).
# - HUQ020: 1=Better, 2=Worse, 3=About the same.
# - HUQ050: numeric count (77/99 treated as 0 in fs3).

paraFile = "paraInit.csv"
paras = pd.read_csv(paraFile, sep=",", header=0)
useDerived = paras.loc[paras["pName"] == "derivedFeatFlag", "pValue"].values[0]

codeBookFile = "codebook_linAge2.csv"
codeBook = pd.read_csv(codeBookFile)

incList = markIncsFromCodeBook(codeBook)

boxCox_lam = pd.read_csv("logNoLog.csv").iloc[1:2, :]


dataMat_trans = pd.read_csv('artifacts/dataMat_trans.csv')
qDataMat = pd.read_csv('artifacts/qDataMat.csv')

vMatDat99_F = pd.read_csv("vMatDat99_F_pre.csv").values
vMatDat99_M = pd.read_csv("vMatDat99_M_pre.csv").values

dataFileName = "mergedDataNHANES9902.csv"
masterData = pd.read_csv(dataFileName)
dataMat = dropCols(masterData, incList) 
qDataMat = qDataMatGen(masterData, incList)

coxCovsTrainM = pd.read_csv('artifacts/coxCovsTrainM.csv')
coxCovsTrainF = pd.read_csv('artifacts/coxCovsTrainF.csv')

# Load
coxModelF = joblib.load("artifacts/cox_full_F.joblib")
nullModelF = joblib.load("artifacts/cox_null_F.joblib")

coxModelM = joblib.load("artifacts/cox_full_M.joblib")
nullModelM = joblib.load("artifacts/cox_null_M.joblib")


CODEBOOK = {
    # demographics (not used in fs-scores, but useful to carry along)
    "SEQN": {"label": "Participant ID (SEQN)", "type": "number", "min": 1, "max": 999999, "value": 1},
    "RIAGENDR": {"label": "Sex", "choices": [("Male", 1), ("Female", 2)], "value": 1},
    "RIDAGEEX": {"label": "Exam Age (years)", "type": "number", "min": 18, "max": 120, "value": 60},

    # disease / conditions (fs1)
    "BPQ020": {"label": "Ever told you had high blood pressure?", "choices": [("Yes",1),("No",2)], "value": 2},
    "DIQ010": {"label": "Doctor told you have diabetes?", "choices": [("Yes",1),("Borderline/Prediabetes",3),("No",2)], "value": 2},
    "KIQ020": {"label": "Ever told you had weak/failing kidneys (excl. stones/cancer)?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ010": {"label": "Ever told you had asthma?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ053": {"label": "Ever told you had emphysema/chronic bronchitis/COPD?", "choices": [("Yes",1),("No",2)], "value": 2},

    # MCQ160* conditions
    "MCQ160A": {"label": "Ever told you had arthritis?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160B": {"label": "(Not used in fs1Score)", "choices": [("No (ignored)",2)], "value": 2},  # explicitly ignored
    "MCQ160C": {"label": "Ever told you had cancer or malignancy?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160D": {"label": "Ever told you had congestive heart failure?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160E": {"label": "Ever told you had coronary heart disease?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160F": {"label": "Ever told you had angina/angina pectoris?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160G": {"label": "Ever told you had heart attack (MI)?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160I": {"label": "Ever told you had stroke?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160J": {"label": "Ever told you had chronic bronchitis?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160K": {"label": "Ever told you had liver condition?", "choices": [("Yes",1),("No",2)], "value": 2},
    "MCQ160L": {"label": "Ever told you had thyroid problem?", "choices": [("Yes",1),("No",2)], "value": 2},

    "MCQ220": {"label": "Ever diagnosed with any other serious illness?", "choices": [("Yes",1),("No",2)], "value": 2},

    # oral health / sleep / function items (fs1)
    "OSQ010A": {"label": "Mouth pain — aching in mouth (past year)?", "choices": [("Yes",1),("No",2)], "value": 2},
    "OSQ010B": {"label": "Mouth pain — tooth sensitive to hot/cold/sweets?", "choices": [("Yes",1),("No",2)], "value": 2},
    "OSQ010C": {"label": "Mouth pain — toothache (past 6 months)?", "choices": [("Yes",1),("No",2)], "value": 2},
    "OSQ060":  {"label": "Difficulty sleeping because of teeth/gums?", "choices": [("Yes",1),("No",2)], "value": 2},

    "PFQ056": {"label": "Any difficulty walking/using steps without equipment?", "choices": [("Yes",1),("No",2)], "value": 2},
    "HUQ070": {"label": "Injured/accident in past 12 months?", "choices": [("Yes",1),("No",2)], "value": 2},

    # self-rated health (fs2)
    "HUQ010": {"label": "General health condition", "choices": [("Excellent",1),("Very good",2),("Good",3),("Fair",4),("Poor",5)], "value": 3},
    "HUQ020": {"label": "Compared with 1 year ago, your health is…", "choices": [("Better",1),("Worse",2),("About the same",3)], "value": 3},

    # healthcare utilization (fs3)
    "HUQ050": {"label": "Times received healthcare (past 12 months)", "type": "number", "min": 0, "max": 99, "value": 0},
}

feature_names = ['BPXPLS', 'BPXSAR', 'BPXDAR', 'BMXBMI', 'URXUMASI', 'URXUCRSI',
       'LBDIRNSI', 'LBDTIBSI', 'LBXPCT', 'LBDFERSI', 'LBDFOLSI', 'LBDB12SI',
       'LBXCOT', 'LBXWBCSI', 'LBXLYPCT', 'LBXMOPCT', 'LBXNEPCT', 'LBXEOPCT',
       'LBXBAPCT', 'LBDLYMNO', 'LBDMONO', 'LBDNENO', 'LBDEONO', 'LBDBANO',
       'LBXRBCSI', 'LBXHGB', 'LBXHCT', 'LBXMCVSI', 'LBXMCHSI', 'LBXMC',
       'LBXRDW', 'LBXPLTSI', 'LBXMPSI', 'LBXCRP', 'LBXGH', 'SSBNP', 'LBDSALSI',
       'LBXSATSI', 'LBXSASSI', 'LBXSAPSI', 'LBDSBUSI', 'LBDSCASI', 'LBXSC3SI',
       'LBDSGLSI', 'LBXSLDSI', 'LBDSPHSI', 'LBDSTBSI', 'LBDSTPSI', 'LBDSUASI',
       'LBDSCRSI', 'LBXSNASI', 'LBXSKSI', 'LBXSCLSI', 'LBDSGBSI', 'fs1Score',
       'fs2Score', 'fs3Score', 'LDLV', 'crAlbRat']

nhanes_desc = {
  # Vitals / anthropometry
  "BPXPLS": "Pulse rate (beats/min, 60-sec pulse).",   # CDC BPX doc
  "BPXSAR": "Systolic blood pressure — average reported to examinee (mmHg).",
  "BPXDAR": "Diastolic blood pressure — average reported to examinee (mmHg).",
  "BMXBMI": "Body mass index (kg/m²).",

  # Kidney (urine)
  "URXUMASI": "Urine albumin (microalbumin), SI units (e.g., mg/L→mg/L; often used for UACR).",
  "URXUCRSI": "Urine creatinine, SI units (mmol/L).",

  # Iron panel
  "LBDIRNSI": "Serum iron (µmol/L).",
  "LBDTIBSI": "Total iron binding capacity, TIBC (µmol/L).",
  "LBXPCT":   "Transferrin saturation (%) = (serum iron / TIBC)×100.",
  "LBDFERSI": "Ferritin (µg/L).",

  # Folate / B12 / cotinine
  "LBDFOLSI": "Serum folate (nmol/L).",
  "LBDB12SI": "Vitamin B12 (pmol/L).",
  "LBXCOT":   "Cotinine (ng/mL)—tobacco exposure marker.",

  # CBC (white cells)
  "LBXWBCSI": "White blood cell count (×10⁹/L).",
  "LBXLYPCT": "Lymphocytes (%).",
  "LBXMOPCT": "Monocytes (%).",
  "LBXNEPCT": "Neutrophils (%).",
  "LBXEOPCT": "Eosinophils (%).",
  "LBXBAPCT": "Basophils (%).",
  "LBDLYMNO": "Lymphocytes (×10⁹/L).",
  "LBDMONO":  "Monocytes (×10⁹/L).",
  "LBDNENO":  "Neutrophils (×10⁹/L).",
  "LBDEONO":  "Eosinophils (×10⁹/L).",
  "LBDBANO":  "Basophils (×10⁹/L).",

  # CBC (red cells / platelets)
  "LBXRBCSI": "Red blood cell count (×10¹²/L).",
  "LBXHGB":   "Hemoglobin (g/dL).",
  "LBXHCT":   "Hematocrit (%).",
  "LBXMCVSI": "Mean corpuscular volume, MCV (fL).",
  "LBXMCHSI": "Mean corpuscular hemoglobin, MCH (pg).",
  "LBXMC":    "Mean corpuscular hemoglobin concentration, MCHC (g/dL).",
  "LBXRDW":   "Red cell distribution width (%).",
  "LBXPLTSI": "Platelet count (×10⁹/L).",
  "LBXMPSI":  "Mean platelet volume, MPV (fL).",

  # Inflammation / glycemia / cardiac
  "LBXCRP": "C-reactive protein (mg/L).",
  "LBXGH":  "Glycohemoglobin (HbA1c, %).",
  "SSBNP":  "N-terminal pro-B-type natriuretic peptide (NT-proBNP, pg/mL).",

  # Basic chem (SI set)
  "LBDSALSI": "Albumin (g/L).",
  "LBXSATSI": "Alanine aminotransferase, ALT (U/L).",
  "LBXSASSI": "Aspartate aminotransferase, AST (U/L).",
  "LBXSAPSI": "Alkaline phosphatase (U/L).",
  "LBDSBUSI": "Urea nitrogen (BUN), SI (mmol/L).",
  "LBDSCASI": "Calcium, SI (mmol/L).",
  "LBXSC3SI": "Bicarbonate (total CO₂), SI (mmol/L).",
  "LBDSGLSI": "Glucose, SI (mmol/L).",
  "LBXSLDSI": "Lactate dehydrogenase, LDH (U/L).",
  "LBDSPHSI": "Phosphorus (mmol/L).",
  "LBDSTBSI": "Total bilirubin (µmol/L).",
  "LBDSTPSI": "Total protein (g/L).",
  "LBDSUASI": "Uric acid (µmol/L).",
  "LBDSCRSI": "Creatinine (µmol/L).",
  "LBXSNASI": "Sodium (mmol/L).",
  "LBXSKSI": "Potassium (mmol/L).",
  "LBXSCLSI": "Chloride (mmol/L).",
  "LBDSGBSI": "Globulin (g/L).",

  # Derived / study-specific
  "fs1Score": "Comorbidity/Frailty index.",
  "fs2Score": "Self-rated health × trajectory.",
  "fs3Score": "Healthcare use (past year).",
  "LDLV":     "Calculated LDL cholesterol (Friedewald or NHANES calc; mg/dL).",
  "crAlbRat": "Urine albumin-to-creatinine ratio (UACR).",
}



def plot_feature_contribs_interactive_np(
    Z_centered: np.ndarray,
    w_feature_years: np.ndarray,
    raw_features: pd.DataFrame,
    feature_names: list[str],
    subject_idx: int = 0,
    title: str | None = None,
    term_age: float | None = None,
    descriptions: dict | pd.Series | None = None,
):
    """
    Z_centered : np.ndarray (n_samples × n_features)
    w_feature_years : np.ndarray (n_features,)
    feature_names : list of feature codes (len = n_features)
    descriptions : dict or pd.Series mapping feature -> human-readable text
    """
    dfZ = pd.DataFrame(Z_centered, columns=feature_names)
    w = pd.Series(np.asarray(w_feature_years), index=feature_names, name="w_years_per_SD")
    row = dfZ.iloc[subject_idx]

    # construct plotting dataframe
    plot_df = pd.DataFrame({
        "feature": feature_names,
        "description": [descriptions.get(f, f) for f in feature_names] if descriptions is not None else feature_names,
        "z_centered": row.values,
        "w_years_per_SD": w.values,
        "contribution_years": (row.values * w.values)/12,
        "lab_values": raw_features[feature_names].iloc[subject_idx].values
    }).sort_values("contribution_years")

    color_continuous_scale=[(0, "blue"), (0.5, "white"), (1, "red")],
    range_color=[plot_df["contribution_years"].min(),
                 plot_df["contribution_years"].max()]

    vals = plot_df["contribution_years"]
    m = float(np.nanquantile(np.abs(vals), 0.9))
    
    # color scale: red→blue (aging/de-aging)
    fig = px.bar(
        plot_df,
        x="contribution_years",
        y="feature",
        orientation="h",
        title=title or f"Feature contributions for subject #{subject_idx}",
        
        color="contribution_years",
        color_continuous_scale="RdBu_r",         # blue for negative, red for positive
        range_color=[-m, m],                     # <-- symmetric
        color_continuous_midpoint=0,
        hover_data=["description", "z_centered", "w_years_per_SD", "contribution_years"],
         width=1600, height=800
    )

    # nice hover tooltip template
    fig.update_traces(
        customdata=plot_df[["description","lab_values","z_centered","w_years_per_SD"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>%{customdata[0]}<br>"
            "user input: %{customdata[1]}<br>"
            "z (centered): %{customdata[2]:.3f}<br>"
            "weight (yrs/SD): %{customdata[3]:.3f}<br>"
            "contrib (yrs): %{x:.3f}<extra></extra>"
        )
    )

    if term_age is not None:
        fig.add_vline(x=float(term_age)/12, line_dash="dash", line_color="black", annotation_text="Age term")

    fig.update_layout(yaxis=dict(dtick=1))
    return fig



# === fs-score calculators (mirror your functions) ===
def fs1Score_from_df(q: pd.DataFrame) -> pd.Series:
    # Safe fills (same as your code)
    def fcol(name, fill):
        s = q[name].copy()
        return s.fillna(fill)

    BPQ020 = fcol("BPQ020", 2); DIQ010 = fcol("DIQ010", 2)
    HUQ010 = fcol("HUQ010", 3); HUQ020 = fcol("HUQ020", 3)
    HUQ050 = fcol("HUQ050", 0); HUQ070 = fcol("HUQ070", 2)
    KIQ020 = fcol("KIQ020", 2); MCQ010 = fcol("MCQ010", 2)
    MCQ053 = fcol("MCQ053", 2); MCQ160A = fcol("MCQ160A", 2)
    MCQ160B = fcol("MCQ160B", 2); MCQ160C = fcol("MCQ160C", 2)
    MCQ160D = fcol("MCQ160D", 2); MCQ160E = fcol("MCQ160E", 2)
    MCQ160F = fcol("MCQ160F", 2); MCQ160G = fcol("MCQ160G", 2)
    MCQ160I = fcol("MCQ160I", 2); MCQ160J = fcol("MCQ160J", 2)
    MCQ160K = fcol("MCQ160K", 2); MCQ160L = fcol("MCQ160L", 2)
    MCQ220 = fcol("MCQ220", 2); OSQ010A = fcol("OSQ010A", 2)
    OSQ010B = fcol("OSQ010B", 2); OSQ010C = fcol("OSQ010C", 2)
    OSQ060  = fcol("OSQ060",  2); PFQ056  = fcol("PFQ056",  2)

    binMat = np.column_stack([
        (BPQ020 == 1), ((DIQ010 == 1) | (DIQ010 == 3)),
        (KIQ020 == 1), (MCQ010 == 1), (MCQ053 == 1),
        (MCQ160A == 1), (MCQ160C == 1), (MCQ160D == 1), (MCQ160E == 1),
        (MCQ160F == 1), (MCQ160G == 1), (MCQ160I == 1), (MCQ160J == 1),
        (MCQ160K == 1), (MCQ160L == 1), (MCQ220 == 1),
        (OSQ010A == 1), (OSQ010B == 1), (OSQ010C == 1), (OSQ060 == 1),
        (PFQ056 == 1), (HUQ070 == 1),
    ])
    return pd.Series(binMat.sum(axis=1) / 22, index=q.index, name="fs1Score")

def fs2Score_from_df(q: pd.DataFrame) -> pd.Series:
    HUQ010 = q["HUQ010"].fillna(3)
    HUQ020 = q["HUQ020"].fillna(3)
    aVec = ((HUQ010 == 4) * 2 + (HUQ010 == 5) * 4)
    dVec = (1 - (HUQ020 == 1) * 0.5 + (HUQ020 == 2))
    return pd.Series(aVec * dVec, index=q.index, name="fs2Score")

def fs3Score_from_df(q: pd.DataFrame) -> pd.Series:
    HUQ050 = q["HUQ050"].copy().fillna(0)
    HUQ050 = HUQ050.replace({77: 0, 99: 0})
    return pd.Series(HUQ050, index=q.index, name="fs3Score")

# === Gradio UI ===
def _radio(label, choices, value):
    return gr.Radio(choices=[lbl for lbl, _ in choices], value=[lbl for lbl, v in choices if v==value][0], label=label)

def _radio_value(radio_value, choices):
    # map label back to numeric code
    for lbl, val in choices:
        if lbl == radio_value: return val
    return None

def build_questionnaire_ui():
    with gr.Column():
        with gr.Accordion("Demographics (optional)", open=False):
            inp_seqn = gr.Number(label=CODEBOOK["SEQN"]["label"], value=CODEBOOK["SEQN"]["value"], precision=0)
            inp_sex  = _radio(CODEBOOK["RIAGENDR"]["label"], CODEBOOK["RIAGENDR"]["choices"], CODEBOOK["RIAGENDR"]["value"])
            inp_age  = gr.Number(label=CODEBOOK["RIDAGEEX"]["label"], value=CODEBOOK["RIDAGEEX"]["value"], precision=0)

        with gr.Accordion("Self-rated health & change (fs2Score)", open=True):
            HUQ010 = _radio(CODEBOOK["HUQ010"]["label"], CODEBOOK["HUQ010"]["choices"], CODEBOOK["HUQ010"]["value"])
            HUQ020 = _radio(CODEBOOK["HUQ020"]["label"], CODEBOOK["HUQ020"]["choices"], CODEBOOK["HUQ020"]["value"])

        with gr.Accordion("Healthcare use (fs3Score)", open=False):
            HUQ050 = gr.Number(label=CODEBOOK["HUQ050"]["label"], value=CODEBOOK["HUQ050"]["value"], precision=0)

        with gr.Accordion("Conditions & limitations (fs1Score)", open=False):
            radios_fs1 = {}
            for k in [
                "BPQ020","DIQ010","KIQ020","MCQ010","MCQ053",
                "MCQ160A","MCQ160C","MCQ160D","MCQ160E","MCQ160F",
                "MCQ160G","MCQ160I","MCQ160J","MCQ160K","MCQ160L",
                "MCQ220","OSQ010A","OSQ010B","OSQ010C","OSQ060",
                "PFQ56","HUQ070"
            ]:
                pass  # we'll add explicitly below to keep labels tidy

            BPQ020 = _radio(CODEBOOK["BPQ020"]["label"], CODEBOOK["BPQ020"]["choices"], CODEBOOK["BPQ020"]["value"])
            DIQ010 = _radio(CODEBOOK["DIQ010"]["label"], CODEBOOK["DIQ010"]["choices"], CODEBOOK["DIQ010"]["value"])
            KIQ020 = _radio(CODEBOOK["KIQ020"]["label"], CODEBOOK["KIQ020"]["choices"], CODEBOOK["KIQ020"]["value"])
            MCQ010 = _radio(CODEBOOK["MCQ010"]["label"], CODEBOOK["MCQ010"]["choices"], CODEBOOK["MCQ010"]["value"])
            MCQ053 = _radio(CODEBOOK["MCQ053"]["label"], CODEBOOK["MCQ053"]["choices"], CODEBOOK["MCQ053"]["value"])

            MCQ160A = _radio(CODEBOOK["MCQ160A"]["label"], CODEBOOK["MCQ160A"]["choices"], CODEBOOK["MCQ160A"]["value"])
            MCQ160B = _radio(CODEBOOK["MCQ160B"]["label"], CODEBOOK["MCQ160B"]["choices"], CODEBOOK["MCQ160B"]["value"])  # not used, kept visible as FYI
            MCQ160C = _radio(CODEBOOK["MCQ160C"]["label"], CODEBOOK["MCQ160C"]["choices"], CODEBOOK["MCQ160C"]["value"])
            MCQ160D = _radio(CODEBOOK["MCQ160D"]["label"], CODEBOOK["MCQ160D"]["choices"], CODEBOOK["MCQ160D"]["value"])
            MCQ160E = _radio(CODEBOOK["MCQ160E"]["label"], CODEBOOK["MCQ160E"]["choices"], CODEBOOK["MCQ160E"]["value"])
            MCQ160F = _radio(CODEBOOK["MCQ160F"]["label"], CODEBOOK["MCQ160F"]["choices"], CODEBOOK["MCQ160F"]["value"])
            MCQ160G = _radio(CODEBOOK["MCQ160G"]["label"], CODEBOOK["MCQ160G"]["choices"], CODEBOOK["MCQ160G"]["value"])
            MCQ160I = _radio(CODEBOOK["MCQ160I"]["label"], CODEBOOK["MCQ160I"]["choices"], CODEBOOK["MCQ160I"]["value"])
            MCQ160J = _radio(CODEBOOK["MCQ160J"]["label"], CODEBOOK["MCQ160J"]["choices"], CODEBOOK["MCQ160J"]["value"])
            MCQ160K = _radio(CODEBOOK["MCQ160K"]["label"], CODEBOOK["MCQ160K"]["choices"], CODEBOOK["MCQ160K"]["value"])
            MCQ160L = _radio(CODEBOOK["MCQ160L"]["label"], CODEBOOK["MCQ160L"]["choices"], CODEBOOK["MCQ160L"]["value"])

            MCQ220 = _radio(CODEBOOK["MCQ220"]["label"], CODEBOOK["MCQ220"]["choices"], CODEBOOK["MCQ220"]["value"])
            OSQ010A = _radio(CODEBOOK["OSQ010A"]["label"], CODEBOOK["OSQ010A"]["choices"], CODEBOOK["OSQ010A"]["value"])
            OSQ010B = _radio(CODEBOOK["OSQ010B"]["label"], CODEBOOK["OSQ010B"]["choices"], CODEBOOK["OSQ010B"]["value"])
            OSQ010C = _radio(CODEBOOK["OSQ010C"]["label"], CODEBOOK["OSQ010C"]["choices"], CODEBOOK["OSQ010C"]["value"])
            OSQ060  = _radio(CODEBOOK["OSQ060"]["label"],  CODEBOOK["OSQ060"]["choices"],  CODEBOOK["OSQ060"]["value"])

            PFQ056 = _radio(CODEBOOK["PFQ056"]["label"], CODEBOOK["PFQ056"]["choices"], CODEBOOK["PFQ056"]["value"])
            HUQ070 = _radio(CODEBOOK["HUQ070"]["label"], CODEBOOK["HUQ070"]["choices"], CODEBOOK["HUQ070"]["value"])

        submit = gr.Button("Compute biological age")

    # return all components to wire in .click
    return {
        # demographics
        "SEQN": inp_seqn, "RIAGENDR": inp_sex, "RIDAGEEX": inp_age,
        # fs2
        "HUQ010": HUQ010, "HUQ020": HUQ020,
        # fs3
        "HUQ050": HUQ050,
        # fs1
        "BPQ020": BPQ020, "DIQ010": DIQ010, "KIQ020": KIQ020, "MCQ010": MCQ010, "MCQ053": MCQ053,
        "MCQ160A": MCQ160A, "MCQ160B": MCQ160B, "MCQ160C": MCQ160C, "MCQ160D": MCQ160D, "MCQ160E": MCQ160E,
        "MCQ160F": MCQ160F, "MCQ160G": MCQ160G, "MCQ160I": MCQ160I, "MCQ160J": MCQ160J, "MCQ160K": MCQ160K, "MCQ160L": MCQ160L,
        "MCQ220": MCQ220, "OSQ010A": OSQ010A, "OSQ010B": OSQ010B, "OSQ010C": OSQ010C, "OSQ060": OSQ060,
        "PFQ056": PFQ056, "HUQ070": HUQ070,
        "submit": submit,
    }

def _collect_values(inputs: dict):
    # map Radio labels back to numeric NHANES codes
    row = {}
    row["SEQN"] = int(inputs["SEQN"])
    row["RIAGENDR"] = dict(CODEBOOK["RIAGENDR"]["choices"])[inputs["RIAGENDR"]]
    row["RIDAGEEX"] = int(inputs["RIDAGEEX"])

    for k in [
        "BPQ020","DIQ010","KIQ020","MCQ010","MCQ053",
        "MCQ160A","MCQ160B","MCQ160C","MCQ160D","MCQ160E","MCQ160F","MCQ160G",
        "MCQ160I","MCQ160J","MCQ160K","MCQ160L","MCQ220",
        "OSQ010A","OSQ010B","OSQ010C","OSQ060","PFQ056","HUQ070",
        "HUQ010","HUQ020"
    ]:
        row[k] = dict(CODEBOOK[k]["choices"])[inputs[k]]

    row["HUQ050"] = int(inputs["HUQ050"])
    return pd.DataFrame([row])

def launch_form():
    with gr.Blocks() as demo:
        # --- UI layout: tabs for readability ---
        with gr.Tabs():
            with gr.Tab("Questionnaire"):
                q_comps = build_questionnaire_ui()   # returns dict with a "submit" button
            with gr.Tab("Labs"):
                lab_comps = build_lab_sliders()      # dict: code -> gr.Slider

        # --- Outputs (add an optional labs echo if you want) ---
        #fig_html = gr.HTML(label="LinAge2 interactive figure")
        fig_html = gr.Plot(label="LinAge2 interactive figure") 

        # Build stable input lists so the click wiring is explicit
        # (avoid relying on dict iteration order)
        q_keys_in_order = [k for k in q_comps.keys() if k != "submit"]
        q_inputs_in_order = [q_comps[k] for k in q_keys_in_order]
        lab_inputs_in_order = [lab_comps[c] for c in LAB_VARIABLES]

        def on_submit(*vals):
            """
            vals = [questionnaire..., labs...], in the order we passed to `inputs=...`.
            """
            n_q = len(q_inputs_in_order)
            q_vals = vals[:n_q]
            lab_vals = vals[n_q:]  # len == len(LAB_VARIABLES)

            # 1) Questionnaire → DataFrame row (your original behavior)
            inp_map = {k: v for k, v in zip(q_keys_in_order, q_vals)}
            qDataMat_user = _collect_values(inp_map)  # must return a 1-row DataFrame or Series compatible with your scoring

            # 2) Optional: attach labs to df (prefixed) so you can persist or feed into inference together
            dataMat_user = pd.DataFrame.from_dict(dict(zip(LAB_VARIABLES, [[x] for x in lab_vals])))
            dataMat_user['SEQN'] = qDataMat_user['SEQN']

            if useDerived:
                print("> Populating derived features ... ", end="")
                print(" fs scores ...", end="")
                
                ######### FS scores
                ## NHANES DATA
                fs1Score = popPCFIfs1(qDataMat)
                fs2Score = popPCFIfs2(qDataMat)
                fs3Score = popPCFIfs3(qDataMat)
                dataMat['fs1Score'] = fs1Score
                dataMat['fs2Score'] = fs2Score
                dataMat['fs3Score'] = fs3Score
                
                ## USER DATA
                fs1Score = popPCFIfs1(qDataMat_user)
                fs2Score = popPCFIfs2(qDataMat_user)
                fs3Score = popPCFIfs3(qDataMat_user)
                dataMat_user['fs1Score'] = fs1Score
                dataMat_user['fs2Score'] = fs2Score
                dataMat_user['fs3Score'] = fs3Score
                
                ######### LDL scores
                ## LDL values
                print(" LDLV ...", end="")
                LDLV = populateLDL(dataMat, qDataMat)
                dataMat['LDLV'] = LDLV
                
                ## USER DATA
                LDLV = populateLDL(dataMat_user, qDataMat_user)
                dataMat_user['LDLV'] = LDLV
                
                ######### Urine albumin to creatinine ratio
                ## Urine Albumin Creatinine ratio
                print(" Albumin Creatinine ratio ... ", end="")
                creaVals = dataMat["URXUCRSI"].values
                albuVals = dataMat["URXUMASI"].values
                crAlbRat = albuVals / (creaVals * 1.1312 * 10**-4)
                dataMat['crAlbRat'] = crAlbRat
                
                ## USER DATA
                creaVals = dataMat_user["URXUCRSI"].values
                albuVals = dataMat_user["URXUMASI"].values
                crAlbRat = albuVals / (creaVals * 1.1312 * 10**-4)
                dataMat_user['crAlbRat'] = crAlbRat

            sex_user = qDataMat_user["RIAGENDR"].values
            initAge_user = qDataMat_user["RIDAGEEX"].values

            dataMat_trans_user = boxCoxTransform(boxCox_lam, dataMat_user)
            dataMatNorm_user = normAsZscores_99_young_mf(dataMat_trans_user.drop(['LBDTCSI', 'LBDHDLSI', 'LBDSTRSI'], axis=1),
                                                         qDataMat_user, dataMat_trans, qDataMat)

            zScoreMax = 6

            dataMatUser_folded = foldOutliers(dataMatNorm_user, zScoreMax)
            
            inputMat_user = dataMatUser_folded.iloc[:, 1:].values
            
            
            sexSel_user = qDataMat_user["RIAGENDR"].values
            
            inputMat_user_M = inputMat_user[sexSel_user == 1, :]
            inputMat_user_F = inputMat_user[sexSel_user == 2, :]
            
            
            
            
            pcMat_user_M = projectToSVD(inputMat_user_M, vMatDat99_M)
            pcMat_user_F = projectToSVD(inputMat_user_F, vMatDat99_F)
            
            
            rowsAll_user = pcMat_user_M.shape[0] + pcMat_user_F.shape[0]
            colsAll = nSVs99_M = 59
            
            pcMat_user = np.zeros((rowsAll_user, colsAll))
            
            
            
            pcMat_user[sexSel_user == 1, :] = pcMat_user_M
            pcMat_user[sexSel_user == 2, :] = pcMat_user_F
            pcMat_user = pd.DataFrame(pcMat_user, columns=[f"PC{i+1}" for i in range(nSVs99_M)])
            
            coxCovs_user = np.column_stack([initAge_user, pcMat_user.values, sex_user])
            coxCovs_user = pd.DataFrame(coxCovs_user, columns=['chronAge'] + list(pcMat_user.columns) + ['sex_user'])
            
            ## Split back into male / female to apply separate models
            coxCovs_user_M = coxCovs_user[sex_user == 1]
            coxCovs_user_F = coxCovs_user[sex_user == 2]

            pc_indices = [int(x[2:])-1 for x in coxModelM.feature_names_in_ if 'PC' in x]

            beta_full = np.zeros(59)
            beta_full[pc_indices] = coxModelM.coef_[1:]
            beta_age_null = nullModelM.coef_[0]
            
            beta_age_full = coxModelM.coef_[0]
            
            w_feature_years = (vMatDat99_M @ beta_full)/beta_age_null
            
            w_age = (beta_age_full / beta_age_null) - 1.0



            mu_PC = np.zeros(59)
            mu_PC[pc_indices] = coxCovsTrainM.mean().loc[coxModelM.feature_names_in_].iloc[1:].values
            mu_age = coxCovsTrainM['chronAge'].mean()
            
            mu_Z = mu_PC@vMatDat99_M.T
            
            Z_centered = inputMat_user_M - mu_Z      # shape (n_samples, n_features)
            term_features = (Z_centered @ w_feature_years)
            term_age = (initAge_user - mu_age) * w_age

            subject_idx=0

            fig = plot_feature_contribs_interactive_np(
            Z_centered,
            w_feature_years,
            dataMat_user,
            feature_names,
            subject_idx=subject_idx,
            title="LinAge2 (M) — feature contributions",
            term_age=float(term_age[subject_idx]) if hasattr(term_age, "__getitem__") else term_age,
            descriptions=nhanes_desc
            )
            html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

            return fig  # lab_dict is just for debugging/inspection

        # Wire the click with explicit ordering of inputs
        q_comps["submit"].click(
            on_submit,
            inputs=q_inputs_in_order + lab_inputs_in_order,
            outputs=fig_html,
        )

    return demo

# If running as a script:
demo = launch_form()
demo.launch(share=True)
