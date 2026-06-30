import numpy as np

class NeuronGroup:
    def __init__(self, size, v_initial=0.0, theta_initial=0.0, 
                 adapt_initial=0.0, refrac_initial=0, silence_initial=0):
        """
        Initialize a group of neurons.
        
        Parameters
        ----------
        size : int
            Number of neurons in the group.
        v_initial : float or array-like, optional
            Initial membrane potential for each neuron. If scalar, all neurons get this value.
        theta_initial : float or array-like, optional
            Initial firing threshold for each neuron. If scalar, all neurons get this value.
        adapt_initial : float or array-like, optional
            Initial adaptation current for each neuron. If scalar, all neurons get this value.
        refrac_initial : int or array-like, optional
            Initial refractory counter for each neuron. If scalar, all neurons get this value.
        silence_initial : int or array-like, optional
            Initial silence counter for each neuron. If scalar, all neurons get this value.
        """
        self.size = size
        # Membrane potential
        self.v = np.full(size, v_initial, dtype=np.float64)
        # Firing threshold
        self.theta = np.full(size, theta_initial, dtype=np.float64)
        # Adaptation current (after-hyperpolarization)
        self.adapt = np.full(size, adapt_initial, dtype=np.float64)
        # Refractory counter (time steps remaining in refractory period)
        self.refrac = np.full(size, refrac_initial, dtype=int)
        # Silence counter (for deprivation-driven recruitment)
        self.silence = np.full(size, silence_initial, dtype=int)

    def apply_leak(self, leak_fraction=0.10, resting=0.0):
        """
        Apply leaky integration to the membrane potential.
        v = v - leak_fraction * (v - resting)
        This is an element-wise operation, no linear algebra.
        """
        self.v = self.v - leak_fraction * (self.v - resting)

    def compute_spikes(self):
        """
        Compute which neurons are above threshold and not refractory.
        Returns a boolean array of shape (size,).
        """
        return (self.v >= self.theta) & (self.refrac == 0)

    def handle_spikes(self, spike_mask, adapt_gain=0.0, refractory_period=0):
        """Handle spiking neurons: reset membrane potential, increment adaptation, set refractory.
        Parameters
        ----------
        spike_mask : boolean array of shape (size,)
            True for neurons that spiked.
        adapt_gain : float
            Amount to add to adaptation for spiking neurons.
        refractory_period : int
            Refractory period to set for spiking neurons.
        """
        # Reset membrane potential for spiking neurons (assuming resting potential is 0.0)
        self.v[spike_mask] = 0.0
        # Increment adaptation current for spiking neurons
        self.adapt[spike_mask] += adapt_gain
        # Set refractory period for spiking neurons
        self.refrac[spike_mask] = refractory_period

    def update_slow_state(self, adapt_decay=0.30, adapt_gain=0.0, 
                          refractory_period=0, silence_threshold=10, 
                          recruit_rate=0.5, winner_idx=None):
        """
        Update slow state variables: adaptation decay, refractory decrement, silence update.
        Parameters
        -----------
        adapt_decay : float
            Fractional decay of adaptation per time step.
        adapt_gain : float
            Amount to add to adaptation if this neuron won (typically ADAPT_GAIN).
        refractory_period : int
            Refractory period to set if this neuron won.
        silence_threshold : int
            Threshold for silence to trigger deprivation-driven recruitment.
        recruit_rate : float
            Rate to increase incoming synapses when silent.
        winner_idx : int or None
            Index of the winning neuron in this group, or None if no winner.
        """
        # Decay adaptation
        self.adapt *= (1.0 - adapt_decay)
        # Add adaptation gain if this neuron won
        if winner_idx is not None:
            self.adapt[winner_idx] += adapt_gain
        # Decrement refractory counters (but not below zero)
        self.refrac = np.maximum(self.refrac - 1, 0)
        # If this neuron won, set its refractory period
        if winner_idx is not None:
            self.refrac[winner_idx] = refractory_period
        # Update silence counters
        if winner_idx is not None:
            # Reset silence for the winner
            self.silence[winner_idx] = 0
            # Increase silence for all others
            mask = np.ones(self.size, dtype=bool)
            mask[winner_idx] = False
            self.silence[mask] += 1
            # Check for deprivation-driven recruitment
            deprived = self.silence > silence_threshold
            if np.any(deprived):
                # For each deprived neuron, we will increase its incoming synapses
                # This is done in the network class by accessing the projections
                pass  # The actual recruitment is handled in the network class

    def get_maturity(self, total_incoming_weights):
        """
        Compute maturity for a neuron given its total incoming weights.
        maturity = total_incoming_weights / MATURITY_THRESHOLD, clipped to [0, 1].
        This is an observable, not used in the leak.
        """
        maturity = total_incoming_weights / 50.0  # MATURITY_THRESHOLD from the model
        if maturity > 1.0:
            maturity = 1.0
        return maturity

    def reset(self, v_initial=0.0, theta_initial=0.0, 
              adapt_initial=0.0, refrac_initial=0, silence_initial=0):
        """
        Reset the neuron group to initial state.
        """
        self.v.fill(v_initial)
        self.theta.fill(theta_initial)
        self.adapt.fill(adapt_initial)
        self.refrac.fill(refrac_initial)
        self.silence.fill(silence_initial)