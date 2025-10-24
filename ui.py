# ---------- codebook_mapping.py ----------
import gradio as gr
import pandas as pd
import numpy as np

from ui_sliders import LAB_VARIABLES, build_lab_sliders

# === Codebook: label + choices (value codes match NHANES) ===
# Notes:
# - Most disease items are 1=Yes, 2=No (default to No).
# - DIQ010 includes 3=Borderline (counts as Yes in fs1).
# - HUQ010: 1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor (fs2 uses Fair/Poor).
# - HUQ020: 1=Better, 2=Worse, 3=About the same.
# - HUQ050: numeric count (77/99 treated as 0 in fs3).
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

        submit = gr.Button("Compute fs1/fs2/fs3")

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
        out_df = gr.Dataframe(label="Questionnaire row (+ optional labs)", interactive=False)
        out_fs = gr.JSON(label="fs-scores")
        out_labs = gr.JSON(label="Labs (ordered dict)", visible=False)  # flip to True to debug

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
            df = _collect_values(inp_map)  # must return a 1-row DataFrame or Series compatible with your scoring

            # 2) Optional: attach labs to df (prefixed) so you can persist or feed into inference together
            lab_dict = dict(zip(LAB_VARIABLES, lab_vals))
            if isinstance(df, pd.DataFrame):
                for k, v in lab_dict.items():
                    df[f"lab__{k}"] = v
            else:
                # if _collect_values returns a Series, coerce to DataFrame
                df = pd.DataFrame([df.to_dict()])
                for k, v in lab_dict.items():
                    df[f"lab__{k}"] = v

            # 3) Compute fs-scores (as before) from questionnaire-only columns
            fs1 = fs1Score_from_df(df)
            fs2 = fs2Score_from_df(df)
            fs3 = fs3Score_from_df(df)
            scores = {
                "fs1Score": float(fs1.iloc[0]),
                "fs2Score": float(fs2.iloc[0]),
                "fs3Score": float(fs3.iloc[0]),
            }

            # 4) Display
            show = df.copy()
            show["fs1Score"] = scores["fs1Score"]
            show["fs2Score"] = scores["fs2Score"]
            show["fs3Score"] = scores["fs3Score"]

            return show, scores, lab_dict  # lab_dict is just for debugging/inspection

        # Wire the click with explicit ordering of inputs
        q_comps["submit"].click(
            on_submit,
            inputs=q_inputs_in_order + lab_inputs_in_order,
            outputs=[out_df, out_fs, out_labs],
        )

    return demo

# If running as a script:
demo = launch_form()
demo.launch(share=True)
