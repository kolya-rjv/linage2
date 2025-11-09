
# Auto-generated Gradio UI sliders for LinAge2 lab measurements (with true min/max)
import gradio as gr

LAB_VARIABLES = ['BPXPLS', 'BPXSAR', 'BPXDAR', 'BMXBMI', 'URXUMASI', 'URXUCRSI', 'LBDIRNSI', 'LBDTIBSI', 'LBXPCT', 'LBDFERSI', 'LBDFOLSI', 'LBDB12SI', 'LBXCOT', 'LBDTCSI', 'LBDHDLSI', 'LBXWBCSI', 'LBXLYPCT', 'LBXMOPCT', 'LBXNEPCT', 'LBXEOPCT', 'LBXBAPCT', 'LBDLYMNO', 'LBDMONO', 'LBDNENO', 'LBDEONO', 'LBDBANO', 'LBXRBCSI', 'LBXHGB', 'LBXHCT', 'LBXMCVSI', 'LBXMCHSI', 'LBXMC', 'LBXRDW', 'LBXPLTSI', 'LBXMPSI', 'LBXCRP', 'LBXGH', 'SSBNP', 'LBDSALSI', 'LBXSATSI', 'LBXSASSI', 'LBXSAPSI', 'LBDSBUSI', 'LBDSCASI', 'LBXSC3SI', 'LBDSGLSI', 'LBXSLDSI', 'LBDSPHSI', 'LBDSTBSI', 'LBDSTPSI', 'LBDSTRSI', 'LBDSUASI', 'LBDSCRSI', 'LBXSNASI', 'LBXSKSI', 'LBXSCLSI', 'LBDSGBSI']

LAB_RANGES = {'BPXPLS': (32.0, 142.0, 1.1, 87.0, 'BPXPLS'), 'BPXSAR': (72.0, 266.0, 1.94, 169.0, 'BPXSAR'), 'BPXDAR': (0.0, 132.0, 1.32, 66.0, 'BPXDAR'), 'BMXBMI': (7.99, 66.44, 0.584, 37.214999999999996, 'BMXBMI'), 'URXUMASI': (0.2, 16920.0, 169.198, 8460.1, 'URXUMASI'), 'URXUCRSI': (177.0, 68422.0, 682.45, 34299.5, 'URXUCRSI'), 'LBDIRNSI': (0.54, 63.9, 0.634, 32.22, 'LBDIRNSI'), 'LBDTIBSI': (5.91, 136.94, 1.31, 71.425, 'LBDTIBSI'), 'LBXPCT': (0.5, 124.2, 1.237, 62.35, 'LBXPCT'), 'LBDFERSI': (1.0, 3234.0, 32.33, 1617.5, 'LBDFERSI'), 'LBDFOLSI': (2.3, 622.9, 6.206, 312.59999999999997, 'LBDFOLSI'), 'LBDB12SI': (27.31, 177562.06, 1775.348, 88794.685, 'LBDB12SI'), 'LBXCOT': (0.011, 1149.0, 11.49, 574.5055, 'LBXCOT'), 'LBDTCSI': (1.84, 18.8, 0.17, 10.32, 'LBDTCSI'), 'LBDHDLSI': (0.21, 4.14, 0.039, 2.175, 'LBDHDLSI'), 'LBXWBCSI': (2.3, 36.6, 0.343, 19.45, 'LBXWBCSI'), 'LBXLYPCT': (4.5, 78.9, 0.744, 41.7, 'LBXLYPCT'), 'LBXMOPCT': (1.0, 33.3, 0.323, 17.15, 'LBXMOPCT'), 'LBXNEPCT': (5.3, 91.3, 0.86, 48.3, 'LBXNEPCT'), 'LBXEOPCT': (0.0, 57.2, 0.572, 28.6, 'LBXEOPCT'), 'LBXBAPCT': (0.0, 35.4, 0.354, 17.7, 'LBXBAPCT'), 'LBDLYMNO': (0.3, 28.9, 0.286, 14.6, 'LBDLYMNO'), 'LBDMONO': (0.1, 4.5, 0.044, 2.3, 'LBDMONO'), 'LBDNENO': (0.3, 17.9, 0.176, 9.1, 'LBDNENO'), 'LBDEONO': (0.0, 5.3, 0.053, 2.65, 'LBDEONO'), 'LBDBANO': (0.0, 4.5, 0.045, 2.25, 'LBDBANO'), 'LBXRBCSI': (1.71, 9.16, 0.074, 5.4350000000000005, 'LBXRBCSI'), 'LBXHGB': (5.9, 19.7, 0.138, 12.8, 'LBXHGB'), 'LBXHCT': (16.3, 59.9, 0.436, 38.1, 'LBXHCT'), 'LBXMCVSI': (51.1, 120.6, 0.695, 85.85, 'LBXMCVSI'), 'LBXMCHSI': (15.0, 43.9, 0.289, 29.45, 'LBXMCHSI'), 'LBXMC': (27.8, 44.9, 0.171, 36.35, 'LBXMC'), 'LBXRDW': (10.5, 31.6, 0.211, 21.05, 'LBXRDW'), 'LBXPLTSI': (11.0, 999.9, 9.889, 505.45, 'LBXPLTSI'), 'LBXMPSI': (5.3, 13.5, 0.082, 9.4, 'LBXMPSI'), 'LBXCRP': (0.01, 29.6, 0.296, 14.805000000000001, 'LBXCRP'), 'LBXGH': (2.5, 18.8, 0.163, 10.65, 'LBXGH'), 'SSBNP': (3.54, 35000.0, 349.965, 17501.77, 'SSBNP'), 'LBDSALSI': (12.0, 57.0, 0.45, 34.5, 'LBDSALSI'), 'LBXSATSI': (3.0, 1925.0, 19.22, 964.0, 'LBXSATSI'), 'LBXSASSI': (7.0, 827.0, 8.2, 417.0, 'LBXSASSI'), 'LBXSAPSI': (12.0, 728.0, 7.16, 370.0, 'LBXSAPSI'), 'LBDSBUSI': (0.36, 43.55, 0.432, 21.955, 'LBDSBUSI'), 'LBDSCASI': (1.725, 3.375, 0.017, 2.55, 'LBDSCASI'), 'LBXSC3SI': (10.0, 38.0, 0.28, 24.0, 'LBXSC3SI'), 'LBDSGLSI': (1.665, 39.246, 0.376, 20.4555, 'LBDSGLSI'), 'LBXSLDSI': (35.0, 719.0, 6.84, 377.0, 'LBXSLDSI'), 'LBDSPHSI': (0.549, 2.615, 0.021, 1.582, 'LBDSPHSI'), 'LBDSTBSI': (1.7, 75.24, 0.735, 38.47, 'LBDSTBSI'), 'LBDSTPSI': (36.0, 110.0, 0.74, 73.0, 'LBDSTPSI'), 'LBDSTRSI': (0.181, 43.512, 0.433, 21.8465, 'LBDSTRSI'), 'LBDSUASI': (23.8, 862.5, 8.387, 443.15, 'LBDSUASI'), 'LBDSCRSI': (8.84, 1140.36, 11.315, 574.5999999999999, 'LBDSCRSI'), 'LBXSNASI': (100.0, 149.6, 0.496, 124.8, 'LBXSNASI'), 'LBXSKSI': (2.5, 6.8, 0.043, 4.65, 'LBXSKSI'), 'LBXSCLSI': (72.0, 114.0, 0.42, 93.0, 'LBXSCLSI'), 'LBDSGBSI': (16.0, 79.0, 0.63, 47.5, 'LBDSGBSI')}

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

sample8881 = {
    'BPXPLS': 68.0, 'BPXSAR': 111.0, 'BPXDAR': 50.0, 'BMXBMI': 31.31,
    'URXUMASI': 96.9, 'URXUCRSI': 18210.0, 'LBDIRNSI': 19.87, 'LBDTIBSI': 54.95,
    'LBXPCT': 36.2, 'LBDFERSI': 81.0, 'LBDFOLSI': 40.5, 'LBDB12SI': 418.45,
    'LBXCOT': 2.0, 'LBDTCSI': 3.83, 'LBDHDLSI': 0.95, 'LBXWBCSI': 6.6,
    'LBXLYPCT': 26.5, 'LBXMOPCT': 8.5, 'LBXNEPCT': 60.1, 'LBXEOPCT': 4.3,
    'LBXBAPCT': 0.6, 'LBDLYMNO': 1.7, 'LBDMONO': 0.6, 'LBDNENO': 4.0,
    'LBDEONO': 0.3, 'LBDBANO': 0.0, 'LBXRBCSI': 4.99, 'LBXHGB': 15.1,
    'LBXHCT': 46.6, 'LBXMCVSI': 93.2, 'LBXMCHSI': 30.3, 'LBXMC': 32.5,
    'LBXRDW': 13.2, 'LBXPLTSI': 293.0, 'LBXMPSI': 8.1, 'LBXCRP': 0.95,
    'LBXGH': 8.0, 'SSBNP': 1029.0, 'LBDSALSI': 43.0, 'LBXSATSI': 15.0,
    'LBXSASSI': 16.0, 'LBXSAPSI': 148.0, 'LBDSBUSI': 3.9, 'LBDSCASI': 2.525,
    'LBXSC3SI': 25.0, 'LBDSGLSI': 3.941, 'LBXSLDSI': 177.0, 'LBDSPHSI': 1.518,
    'LBDSTBSI': 13.7, 'LBDSTPSI': 79.0, 'LBDSTRSI': 1.027, 'LBDSUASI': 303.3,
    'LBDSCRSI': 79.6, 'LBXSNASI': 142.4, 'LBXSKSI': 4.35, 'LBXSCLSI': 101.6,
    'LBDSGBSI': 36.0
}

def patch_lab_defaults(LAB_RANGES, sample_dict):
    new_ranges = {}
    for var, tpl in LAB_RANGES.items():
        mn, mx, step, default, label = tpl
        val = sample_dict.get(var, default)
        # clamp to slider bounds to avoid UI errors
        if isinstance(val, (int, float)):
            val = float(min(max(val, mn), mx))
        new_ranges[var] = (mn, mx, step, val, label)
    return new_ranges

#LAB_RANGES = patch_lab_defaults(LAB_RANGES, sample8881)

def lab_slider_for(code):
    lo, hi, step, default, label = LAB_RANGES[code]
    # row: [Slider | Missing □]
    with gr.Row():
        sld = gr.Slider(
            minimum=lo, maximum=hi, step=step, value=default,
            label=f"{nhanes_desc.get(label, '')} ({label})", interactive=False
        )
        chk = gr.Checkbox(label="Missing", value=True, scale=0)  # default missing=True
    # when Missing is checked -> disable slider
    def _toggle(missing):
        return gr.update(interactive=not missing)
    chk.change(_toggle, inputs=chk, outputs=sld)
    return sld, chk

def build_lab_sliders():
    comps = {}
    with gr.Accordion("Lab Inputs (NHANES codes)", open=False):
        # bulk controls
        with gr.Row():
            btn_all_missing = gr.Button(value="Mark all Missing")
            btn_all_known   = gr.Button(value="Mark all Known")
        cols = [gr.Column(), gr.Column(), gr.Column()]
        missing_boxes = []
        sliders = []
        for i, code in enumerate(LAB_VARIABLES):
            with cols[i % 3]:
                sld, chk = lab_slider_for(code)
                comps[code] = {"slider": sld, "missing": chk}
                sliders.append(sld)
                missing_boxes.append(chk)
        # wire bulk buttons
        def _set_all_missing(_):
            return [gr.update(value=True) for _ in missing_boxes] + \
                   [gr.update(interactive=False) for _ in sliders]
        def _set_all_known(_):
            return [gr.update(value=False) for _ in missing_boxes] + \
                   [gr.update(interactive=True) for _ in sliders]
        btn_all_missing.click(_set_all_missing, inputs=[], outputs=missing_boxes + sliders)
        btn_all_known.click(_set_all_known, inputs=[], outputs=missing_boxes + sliders)
    return comps

def read_lab_values(components_dict):
    vals = []
    for code in LAB_VARIABLES:
        sld = components_dict[code]["slider"]
        miss = components_dict[code]["missing"].value

        if miss:
            vals.append(999)
            continue

        v = sld.value
        # Fail fast if UI violated the contract
        assert v is not None, f"{code}: slider has no value when marked known"
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"{code}: expected numeric slider value, got {v!r}")
    return vals

if __name__ == '__main__':
    with gr.Blocks(title="LinAge2 — Lab Inputs") as demo:
        gr.Markdown("## LinAge2 — Laboratory Measurements")
        comps = build_lab_sliders()
        btn = gr.Button("Preview Values")
        out = gr.JSON(label="Slider Values")
        def _collect(*vals):
            return dict(zip(LAB_VARIABLES, vals))
        btn.click(fn=_collect, inputs=[comps[c] for c in LAB_VARIABLES], outputs=out)
    demo.launch(quiet=True)
