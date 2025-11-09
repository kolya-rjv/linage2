import pandas as pd

from src import nonAccidDeathFlags
from ui_sliders import LAB_VARIABLES


dataFileName = "mergedDataNHANES9902.csv"
masterData = pd.read_csv(dataFileName)

imputation_pool = masterData[nonAccidDeathFlags(masterData)].query('yearsNHANES == 9900')

def count_reference_values(sex, age, offset_years=5):
    """
    Compute median reference (imputation) values for a given sex and age window.

    This function selects a subset of the global `imputation_pool` DataFrame
    matching the specified sex and age range, and returns median values for
    all numeric columns. The age window is defined as ±`offset_years` around
    the target age (in years), but truncated at the 40-year boundary so that
    imputation for younger users (<40) never includes samples ≥40, and vice versa.

    Parameters
    ----------
    sex : int
        NHANES RIAGENDR code (1 = male, 2 = female).
    age : float
        Age in months for the user or input sample.
    offset_years : int, optional
        Half-width of the age window (in years) used for selecting reference
        samples. Default is 5 years.

    Returns
    -------
    dict
        Dictionary of median values for all numeric columns in the selected
        reference subset. If no samples fall within the age window (a rare
        case), medians are computed from all rows of the same sex.
    """
    age_years = age / 12

    if age_years < 40:
        low = age_years - offset_years
        high = min(age_years + offset_years, 40)
    else:
        low = max(age_years - offset_years, 40)
        high = age_years + offset_years

    low *= 12
    high *= 12
    imputation_bin = imputation_pool.query(f'RIAGENDR == {sex} & RIDAGEEX >= {low} & RIDAGEEX <= {high}')
    if not imputation_bin.shape[0]:  # very rare: no samples in age window
        imputation_bin = imputation_pool.query(f'RIAGENDR == {sex}')

    ref_values = imputation_bin.median(skipna=True)
    return ref_values.to_dict()


def impute_missing_values(raw_vals, flags, sex, age):
    """
    raw_vals: list of floats (or None) from sliders
    flags: list of booleans; True => missing
    sex: NHANES RIAGENDR (1/2)
    age: age in months (RIDAGEEX)
    """
    ref_dict = count_reference_values(sex=sex, age=age, offset_years=5)
    # fill per LAB_VARIABLES order; fall back to 999 if a code isn't in ref_dict (shouldn’t happen)
    return [
        (v if not flag else float(ref_dict[code]))
        for code, v, flag in zip(LAB_VARIABLES, raw_vals, flags)
    ]
