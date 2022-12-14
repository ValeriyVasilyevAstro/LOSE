class Constants:
    cadence: float = 30.0/60/24.0
    number_of_cadences_in_one_hour_lc: int = 2


class Parameters:
    number_of_excluded_cadences_before_flare: int = 2
    number_of_excluded_cadences_after_flare: int = 4

    n_steps_default: int = 12000
    n_discard_default: int = 8000

    window_length_hours: float = 16.5

    order_of_the_fitted_polynomial: int = 3