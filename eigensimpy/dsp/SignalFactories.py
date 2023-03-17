import numpy as np
from eigensimpy.dsp.Signals import Signal, Quantity, DimensionArray, Dimension


class PulseFactory:

    def __init__(self, **kwargs) -> None:
        self.center_frequency = kwargs.get('center_frequency', 7e6)
        self.sampling_freq = kwargs.get('sampling_freq', 65e6)
        self.bandwidth = kwargs.get('bandwidth', 0.5)
        self.cutoff = kwargs.get('cutoff', -35)
        self.bwref = kwargs.get('bwref', -6)
        
    def calc_time_vector(self) -> np.ndarray:

        t0 = self.calc_t0()
        t = np.arange(-t0, t0, 1 / self.sampling_freq)
        # make a column vector, -1 means automatically determine num rows 
        t = t.reshape(-1, 1)
        return t

    def calc_t0(self) -> float:

        fc = self.center_frequency
        bw = self.bandwidth
        cf = self.cutoff
        bwr = self.bwref

        rlevel = 10 ** (bwr / 20)
        fv = (-bw ** 2 * fc ** 2) / (8 * np.log(rlevel))
        tvar = (4 * np.pi ** 2 * fv) ** -1

        d = 10 ** (cf / 20)
        t0 = np.sqrt(-2 * tvar * np.log(d))

        return t0

    def calc_pulse(self, pulse_type: str) -> Signal:

        t = self.calc_time_vector()
        dt = 1 / self.sampling_freq

        fc = self.center_frequency
        bw = self.bandwidth
        beta = fc * bw

        if pulse_type == 'gaussian':
            pulse = np.multiply(np.exp(-(((np.pi * bw) ** 2) * (t ** 2))), np.cos(2 * np.pi * fc * t)).flatten()

        elif pulse_type == 'modray':
            t = t - np.min(t)
            pulse = t * np.sin(2 * np.pi * fc * t) * np.exp(-4 * beta ** 2 * t ** 2)

        elif pulse_type == 'simple':
            alpha = 4.44
            pulse = -(t) * np.exp(-(alpha * fc * (t)) ** 2)

        elif pulse_type == 'simple1':
            alpha = 3.116
            pulse = -(t) * np.exp(-(alpha * fc * (t)) ** 2)
            pulse = np.diff(pulse, axis=0)

        elif pulse_type == 'simple2':
            alpha = 2.5
            pulse = -(t) * np.exp(-(alpha * fc * (t)) ** 2)
            pulse = np.diff(np.diff(pulse, axis=0), axis=0)

        elif pulse_type == 'simple3':
            alpha = 2.15
            pulse = -(t) * np.exp(-(alpha * fc * (t)) ** 2)
            pulse = np.diff(np.diff(np.diff(np.diff(pulse, axis=0), axis=0), axis=0), axis=0)
            pulse = pulse / np.max(pulse) * -1

        elif pulse_type == 'simple4':
            alpha = 2.15
            pulse = -(t) * np.exp(-(alpha * fc * (t)) ** 2)
            pulse = np.diff(np.diff(np.diff(np.diff(pulse, axis=0), axis=0), axis=0), axis=0)

        elif pulse_type == 'simple5':
            alpha = 4
            pulse = alpha = 4
            pulse = np.sin(2 * np.pi * fc * t) * np.exp(-alpha * ((t * fc - 1) ** 2))

        else:
            raise ValueError("Unknown Type")

        t = t - np.min(t)
        pulse = pulse.reshape(-1, 1)
        pulse = pulse / np.max(pulse)

        if len(pulse) != len(t):
            t = t[:len(pulse)]

        # convert to microseconds
        t = t * 1e6
        dt = t[1] - t[0]

        qu = Quantity(name='Time', si_unit='μs')
        
        dim = DimensionArray([ Dimension( quantity=qu, delta=dt, offset=t[0]), Dimension()] )
        sig = Signal(data=pulse, dims=dim)

        return sig