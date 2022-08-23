from lightkurve.targetpixelfile import TargetPixelFile
from src.image_processing import FlareImageComputation, FlareImageComputation_Input, FlareImageComputation_Output
from src.mcmc import MCMCFitting, MCMCFitting_Input, MCMCFitting_Output

class FlareLocalization:
    def __init__(self, flare_time: float, tpf: TargetPixelFile, window_length_hours: float = 16.5):
        self.flare_time = flare_time
        self.tpf = tpf
        self.window_length_hours = window_length_hours
        self.flare_image: FlareImageComputation_Output = None
        self.mcmc = None

    def compute_flare_image(self) -> "FlareImageComputation_Output":
        self.flare_image: FlareImageComputation_Output = \
            FlareImageComputation().processing(inp=FlareImageComputation_Input(
                tpf=self.tpf,flare_time=self.flare_time, window_length_hours=self.window_length_hours))
        return self

    def localize_flare(self, n_steps_default, n_discard_default) -> "FlareImageComputation_Output":
        self.mcmc: MCMCFitting_Output = MCMCFitting().run(inp=MCMCFitting_Input(
            flare_image=self.flare_image.flare_image, tpf=self.tpf, n_steps=n_steps_default, n_discard=n_discard_default))
        return self




