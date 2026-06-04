METADATA_COLS = [
    "id", "sample_type", "group", "group_fam", "timepoint", "person_code",
    "sample_code", "age_years", "height_cm",
    # Regression targets below — all excluded from FTIR features
    "bodyweight_kg", "bodyfat_kg", "bodyfat_perc", "ffm_kg", "ffm_%",
    "h2o_L", "h20_perc",
    "anaerobicthreshold_relative", "anaerobicthreshold_bpm", "anaerobicthreshold_absolute",
    "respiratorycompensation_relative", "respiratorycompensation_bpm", "respiratorycompensation_absolute",
    "vo2max_relative", "vo2max_bpm", "vo2max_absolute",
    "erythrocytes", "hemoglobine", "hematocrit", "mgv", "mch", "mchc", "rdw",
    "leukocytes", "neutrophiles", "eosinophils", "basophils", "lymphocytes", "monocytes", "platetes",
    "salivarycortisol", "salivarytestosterone", "il_10",
    "vo2max_classes", "vo2max_classes_simplified",
    "bodyfat_classes", "bodyfat_classes_simplified",
    "path_length",
]

SAMPLE_TYPES = ["CAPILAR", "PLASMA", "SALIVA", "SERUM", "URINE"]

WATER_REGION = (1850.0, 2500.0)  # cm⁻¹ — excluded from FTIR analysis

TARGET_GROUPS = {
    "cardiorespiratory": [
        "vo2max_absolute", "vo2max_relative", "vo2max_bpm",
        "anaerobicthreshold_absolute", "anaerobicthreshold_relative", "anaerobicthreshold_bpm",
        "respiratorycompensation_absolute", "respiratorycompensation_relative", "respiratorycompensation_bpm",
    ],
    "body_composition": [
        "bodyweight_kg", "bodyfat_kg", "bodyfat_perc",
        "ffm_kg", "ffm_%", "h2o_L", "h20_perc",
    ],
    "cbc": [
        "erythrocytes", "hemoglobine", "hematocrit", "mgv", "mch", "mchc", "rdw",
        "leukocytes", "neutrophiles", "eosinophils", "basophils", "lymphocytes",
        "monocytes", "platetes",
    ],
    "hormonal": [
        "salivarycortisol", "salivarytestosterone", "il_10",
    ],
}

REGRESSION_TARGETS = [t for targets in TARGET_GROUPS.values() for t in targets]


def get_target_group(target: str) -> str:
    for group, targets in TARGET_GROUPS.items():
        if target in targets:
            return group
    return "other"
