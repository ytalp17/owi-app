import pandas

GENDER = "men"

event_weights = pandas.read_csv("event_weights.csv")
athlete_countries = pandas.read_csv('app_data/' + GENDER + "/athlete_countries.csv")
RESULTS_DIRECTORY = 'app_data/' + GENDER + "/results"
SPLITS_DIRECTORY = 'app_data/' + GENDER + "/splits"
RANKINGS_DIRECTORY = 'app_data/' + GENDER + "/rankings_archive"
ATHLETE_DATA_DIRECTORY = 'app_data/' + GENDER + "/athlete_data"
RANKING_FILE_NAME = "PageRanking.csv"

# Depreciation Period: time period in days over which a depreciation is applied to the initial weight of a result.
W_DEPRECIATION_PERIOD = 365 * 3.427083333
M_DEPRECIATION_PERIOD = 365 * 3.65625

# Choose age_weight depreciation curve type: "linear", "exponential", or "sigmoid"
DEPRECIATION_MODEL = "sigmoid"

# Drives age_weight_exp() exponential decay function. The more negative, the quicker the decline in age_weight.
LAMBDA = -1.4

# Steepness of sigmoid depreciation curve. 1 = linear, > 1 increases steepness. Cannot be < 1.
W_K = 1.25
M_K = 2.5

# Tune ranking for the following distance in km. 0 for no preference, or best "overall" regardless of dist.
RANK_DIST = 10

# Predictability settings: optimize test for a subset of the overall ranking.
FROM_RANK = 1
TO_RANK = 2000

# Color


# Theme
THEME = { 
        "colorScheme": "dark",
        "fontFamily": "'Inter', sans-serif",
        "primaryColor": "#3BC9DB",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
        },
    }


