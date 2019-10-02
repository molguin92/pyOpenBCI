import threading
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from pyOpenBCI import OpenBCIGanglion

NOMINAL_SAMPLING_RATE = 200


class GanglionSampleCounter:
    def __init__(self, board: OpenBCIGanglion):
        self._samples = 0
        self._lost_samples = 0
        self._board = board
        self._sampling_thread = threading.Thread(
            target=GanglionSampleCounter._recv_samples,
            args=(self,),
            daemon=True)

        self._lock = threading.RLock()

    def start(self):
        self._sampling_thread.start()

    def stop(self):
        self._board.stop_stream()
        self._sampling_thread.join()

    def _recv_samples(self):
        self._board.start_stream(self.handle_sample)

    @property
    def sample_count(self) -> Tuple[int, int]:
        with self._lock:
            return self._samples, self._lost_samples

    def handle_sample(self, sample):
        with self._lock:
            if np.NaN not in sample.channels_data:
                self._samples += 1
            else:
                self._lost_samples += 1


def benchmark_ganglion(mac: str,
                       runs: int, run_duration: float) -> np.ndarray:
    sampling_rates = []
    board = OpenBCIGanglion(mac=mac)

    total_recv_samples = 0
    total_lost_samples = 0

    counter = GanglionSampleCounter(board)
    counter.start()

    print(f'Sleeping for 10 seconds to let receiver warm up.')
    time.sleep(10.0)

    for r in range(runs):
        print(f'\rExecuting run {r + 1}/{runs}...', end='', flush=True)
        samples_i, lost_samples_i = counter.sample_count
        t_i = time.time()
        time.sleep(run_duration)
        delta_t = time.time() - t_i

        samples, lost_samples = counter.sample_count
        delta_cnt = samples - samples_i

        total_recv_samples += delta_cnt
        total_lost_samples += lost_samples - lost_samples_i

        sampling_rates.append(delta_cnt / delta_t)

    sampling_rates = np.array(sampling_rates)
    _, minmax, mean, var, _, _, = scipy.stats.describe(sampling_rates)

    counter.stop()
    board.disconnect()

    print(f'''
... Sampling rate benchmark results for OpenBCI Ganglion ...
Measured sampling rates over ~{run_duration:.2f}s intervals:
Mean: {mean} Hz
Min: {minmax[0]} Hz
Max: {minmax[1]} Hz
Variance: {var} Hz

Total received samples: {total_recv_samples}
Total lost samples: {total_lost_samples}
Loss rate: {(total_lost_samples / (total_recv_samples + total_lost_samples))
            * 100.0}%
    ''')

    return sampling_rates


if __name__ == '__main__':
    data = benchmark_ganglion('D2:EA:16:D2:EB:3F', 300, 1.0)

    fig, ax = plt.subplots()
    ax.axhline(y=200)
    ax.boxplot([data], vert=True, labels=['EEG\nNom. Freq. 200Hz'])

    y_ticks = ax.get_yticks()
    ax.set_yticks(np.append(y_ticks, np.array([200])))
    ax.legend()
    ax.set_ylabel('Measured sampling frequency [Hz]')
    plt.show()
