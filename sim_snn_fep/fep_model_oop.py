    def _recompute_maturity_l3(self, j):
        """maturity for L3 layer: how close a single volley of L2->L3 gates is to theta."""
        total = 0.0
        for i in range(self.L2E.size):
            total += self.proj_L2E_L3E.weights[j, i]
        maturity = total / MATURITY_THRESHOLD
        if maturity > 1.0:
            maturity = 1.0
        return maturity