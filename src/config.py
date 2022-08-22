class Constants:
    cadence: float = 30.0/60/24.0
    number_of_cadences_in_one_hour_lc: int = 2


class Parameters:
    number_of_excluded_cadences_before_flare: int = 2
    number_of_excluded_cadences_after_flare: int = 4