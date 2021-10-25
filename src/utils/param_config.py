class HParams:
    """
    Class of HParam for claibration, exploration, estimation
    """
    def __init__(self):
        # calibration
        self.calibration = {}
        
        # exploration
        self.exploration = {}
        
        # estimation
        self.estimation = {'gp_laplacian_count': 0.}