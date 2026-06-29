import numpy as np
from fep_model import FEPSNN

snn = FEPSNN()
# set adapt to zero so v_l2e = 0
snn.adapt[:] = 0.0
snn.refrac[:] = 0
snn.silence[:] = 0
# manually override v_l2e to 100 by setting adapt = -100
snn.adapt[:] = -100.0
print("Initial v_l2e:", snn.v_l2e)
# run one presentation with empty pattern
snn.process_event_pattern([])
print("After empty pattern v_l2e:", snn.v_l2e)
print("Adapt after:", snn.adapt)