from lightkurve.targetpixelfile import TargetPixelFile
from src.image_processing import FlareImageComputation, FlareImageComputation_Input, FlareImageComputation_Output
from src.mcmc import MCMCFitting, MCMCFitting_Input, MCMCFitting_Output
from src.config import Parameters
from src.get_gaia_data import StarPositionsFromGAIA

class FlareLocalization:
    def __init__(self, flare_time: float, tpf: TargetPixelFile, window_length_hours: float = Parameters.window_length_hours):
        self.flare_time = flare_time
        self.tpf = tpf
        self.window_length_hours = window_length_hours
        self.flare_image: FlareImageComputation_Output = None
        self.mcmc = None
        self.star_positions = None
        self.gaia_data = None

    def compute_flare_image(self) -> "FlareImageComputation_Output":
        self.flare_image: FlareImageComputation_Output = \
            FlareImageComputation().processing(inp=FlareImageComputation_Input(
                tpf=self.tpf,flare_time=self.flare_time, window_length_hours=self.window_length_hours))
        return self

    def localize_flare(self, n_steps: int = Parameters.n_steps_default, n_discard: int = Parameters.n_discard_default) -> "FlareImageComputation_Output":
        self.mcmc: MCMCFitting_Output = MCMCFitting().run(inp=MCMCFitting_Input(
            flare_image=self.flare_image.flare_image, tpf=self.tpf, n_steps=n_steps, n_discard=n_discard))
        return self

    def get_gaia_data(self):
        self.star_positions, self.gaia_data = StarPositionsFromGAIA().get_gaia_data(tpf_object=self.tpf)
        return self




