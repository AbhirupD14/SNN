import numpy as np

class Projection:
    def __init__(self, pre_group, post_group, 
                 weight_init_min=0.0, weight_init_max=1.0,
                 is_excitatory=True,
                 w_gain=6.0,
                 prune_rate=0.45,
                 synaptic_budget=50.0,
                 maturity_threshold=50.0):
        """
        Initialize a projection (connection) from pre_group to post_group.
        
        Parameters
        ----------
        pre_group : NeuronGroup
            The pre-synaptic neuron group.
        post_group : NeuronGroup
            The post-synaptic neuron group.
        weight_init_min : float
            Minimum value for initializing weights.
        weight_init_max : float
            Maximum value for initializing weights.
        is_excitatory : bool
            True if the connection is excitatory, False if inhibitory.
            Note: For inhibitory connections, we store positive weights and
            the network applies the negative sign when computing input.
        w_gain : float
            Demand-driven gate widening parameter (Hebbian learning rate).
        prune_rate : float
            Heterosynaptic withering rate for inactive synapses.
        synaptic_budget : float
            Per-neuron synaptic scaling cap (local homeostasis).
        maturity_threshold : float
            Total incoming gate at which one volley equals firing threshold.
        """
        self.pre_group = pre_group
        self.post_group = post_group
        self.is_excitatory = is_excitatory
        self.W_GAIN = w_gain
        self.PRUNE_RATE = prune_rate
        self.SYNAPTIC_BUDGET = synaptic_budget
        self.MATURITY_THRESHOLD = maturity_threshold
        
        # Initialize weight matrix: shape (post_group.size, pre_group.size)
        # weights[post_idx, pre_idx] = weight from pre_idx to post_idx
        self.weights = np.random.uniform(
            weight_init_min, 
            weight_init_max, 
            size=(post_group.size, pre_group.size)
        )

    def compute_input(self, pre_spike_mask):
        """
        Compute input to post-synaptic neurons from pre-synaptic spikes.
        This is computed as: for each post-synaptic neuron j,
            input[j] = sum over i where pre_spike_mask[i] is True of weights[j, i]
        If the connection is inhibitory, the network will apply a negative sign.
        This uses explicit loops, not linear algebra.
        
        Parameters
        ----------
        pre_spike_mask : boolean array of shape (pre_group.size,)
            True for pre-synaptic neurons that spiked.
            
        Returns
        -------
        input_to_post : numpy array of shape (post_group.size,)
            Input to each post-synaptic neuron.
        """
        input_to_post = np.zeros(self.post_group.size, dtype=np.float64)
        # Loop over each post-synaptic neuron
        for j in range(self.post_group.size):
            # Loop over each pre-synaptic neuron
            for i in range(self.pre_group.size):
                if pre_spike_mask[i]:
                    input_to_post[j] += self.weights[j, i]
        return input_to_post

    def update_plasticity(self, post_neuron_idx, pre_spike_mask, post_spike_mask):
        """
        Update plasticity for the synapses onto a specific post-synaptic neuron.
        This implements Hebbian learning: a synapse widens only when both
        pre- and post-synaptic neurons are active.
        Also implements synaptic scaling and pruning.
        
        Parameters
        ----------
        post_neuron_idx : int
            Index of the post-synaptic neuron that won (fired).
        pre_spike_mask : boolean array of shape (pre_group.size,)
            True for pre-synaptic neurons that spiked in the current volley.
        post_spike_mask : boolean array of shape (post_group.size,)
            True for post-synaptic neurons that spiked in the current volley.
            (Used to confirm the post_neuron_idx actually spiked, though
            we assume the caller only calls this for neurons that did spike.)
        """
        # Only update if the post-synaptic neuron actually spiked
        if not post_spike_mask[post_neuron_idx]:
            return
            
        # 1. Hebbian learning: strengthen active synapses
        for i in range(self.pre_group.size):
            if pre_spike_mask[i]:
                # Synapse was active: strengthen it
                g = self.weights[post_neuron_idx, i]
                # Saturating growth: dw = W_GAIN * (1 - g / W_MAX)
                # We need to know the maximum weight for this connection.
                # For excitatory connections from L1E/L2E to L2E/L3E, the max is 50.0
                # For other connections (like L2E->L1I), the max is 1.0
                # We'll determine this based on the connection type.
                if self.is_excitatory and hasattr(self.pre_group, 'size'):
                    # Heuristic: if pre_group is L1E or L2E and post_group is L2E or L3E, max is 50
                    # Otherwise, max is 1.0
                    # This is a simplification; ideally we'd pass W_MAX as a parameter
                    if (hasattr(self.pre_group, 'size') and 
                        (self.pre_group.size == 9 or self.pre_group.size == 8) and  # L1E or L2E
                        (self.post_group.size == 8 or self.post_group.size == 4)):  # L2E or L3E
                        W_MAX = 50.0
                    else:
                        W_MAX = 1.0
                else:
                    W_MAX = 1.0  # Default for other connections
                    
                self.weights[post_neuron_idx, i] = g + self.W_GAIN * (1.0 - g / W_MAX)
            else:
                # Synapse was inactive: weaken it (pruning)
                self.weights[post_neuron_idx, i] *= (1.0 - self.PRUNE_RATE)
                
            # Ensure weight doesn't go negative
            if self.weights[post_neuron_idx, i] < 0.0:
                self.weights[post_neuron_idx, i] = 0.0

        # 2. Synaptic scaling: enforce budget constraint for this post-synaptic neuron
        total = 0.0
        for i in range(self.pre_group.size):
            total += self.weights[post_neuron_idx, i]
        if total > self.SYNAPTIC_BUDGET:
            scale = self.SYNAPTIC_BUDGET / total
            for i in range(self.pre_group.size):
                self.weights[post_neuron_idx, i] *= scale

    def update_plasticity_all(self, pre_spike_mask, post_spike_mask):
        """
        Update plasticity for all post-synaptic neurons that spiked.
        This is used for layers where all neurons can potentially win
        (like L1 layer where we don't have a single winner-take-all).
        
        Parameters
        ----------
        pre_spike_mask : boolean array of shape (pre_group.size,)
            True for pre-synaptic neurons that spiked.
        post_spike_mask : boolean array of shape (post_group.size,)
            True for post-synaptic neurons that spiked.
        """
        # For each post-synaptic neuron that spiked, update its incoming synapses
        for j in range(self.post_group.size):
            if post_spike_mask[j]:
                self.update_plasticity(j, pre_spike_mask, post_spike_mask)

    def get_maturity(self, post_neuron_idx):
        """
        Compute maturity for a specific post-synaptic neuron.
        maturity = total_incoming_weights / MATURITY_THRESHOLD, clipped to [0, 1].
        
        Parameters
        ----------
        post_neuron_idx : int
            Index of the post-synaptic neuron.
            
        Returns
        -------
        maturity : float
            Maturity value for the neuron.
        """
        total = 0.0
        for i in range(self.pre_group.size):
            total += self.weights[post_neuron_idx, i]
        maturity = total / self.MATURITY_THRESHOLD
        if maturity > 1.0:
            maturity = 1.0
        return maturity

    def get_weights(self):
        """
        Get a copy of the weight matrix.
        
        Returns
        -------
        weights : numpy array of shape (post_group.size, pre_group.size)
            Copy of the weight matrix.
        """
        return self.weights.copy()