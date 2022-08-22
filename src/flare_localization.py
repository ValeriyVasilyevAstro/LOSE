from lightkurve.targetpixelfile import TargetPixelFile
from src.image_processing import FlareImageComputation, FlareImageComputation_Input, FlareImageComputation_Output


class FlareLocalization:
    def __init__(self, flare_time: float, tpf: TargetPixelFile, window_length_hours: float = 16.5):
        self.flare_time = flare_time
        self.tpf = tpf
        self.flare_image_computation: FlareImageComputation_Output = None
        self.window_length_hours = window_length_hours

    def compute_flare_image(self):
        self.flare_image = FlareImageComputation().processing(inp=FlareImageComputation_Input(tpf=self.tpf,
                                                                                              flare_time=self.flare_time,
                                                                                              window_length_hours=self.window_length_hours))




