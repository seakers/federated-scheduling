import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

class FederatedSatelliteEnv:
    """
    Simulates the true, hidden competitive environment.
    The broker has no direct visibility into this class.
    """
    def __init__(self):
        # 3x3 Grid: Rows are Latitude Bands, Cols are Longitude Bands
        # True hidden acceptance probabilities (P_accept) for our broker's bids
        self.true_probs = np.array([
            [0.15, 0.60, 0.95],  # Row 0: Hotspot (e.g., ME), Standard, Ocean (e.g., Pac)
            [0.60, 0.60, 0.60],  # Row 1: Standard zones
            [0.95, 0.60, 0.15]   # Row 2: Ocean, Standard, Hotspot
        ])
        self.drift_triggered = False

    def update_environment(self, epoch):
        # Trigger sudden non-stationarity at epoch 500
        # A natural disaster/crisis in the Pacific Ocean cell (0, 2) causes competitor traffic to spike.
        # True acceptance probability crashes from 0.95 down to 0.10.
        if epoch >= 500 and not self.drift_triggered:
            self.true_probs[0, 2] = 0.10  # Apply the crash specifically to the Pacific Zone
            self.drift_triggered = True
            print(f"\n[Epoch {epoch}] CRISIS TRIGGERED in Pacific Zone (0, 2)!")
            print("Competitor traffic has surged. True acceptance probability crashed from 0.95 to 0.10.\n")
        return self.true_probs

    def generate_feedback(self, row, col):
        # Simulate binary clearing outcome based on the true hidden probability
        prob = self.true_probs[row, col]
        return np.random.binomial(1, prob)


class ThompsonSchedulingBroker:
    def __init__(self, use_heuristics=True, gamma=1.0, size=(3, 3)):
        self.size = size
        self.gamma = gamma
        self.use_heuristics = use_heuristics
        
        # Initialize parameters with Beta(1,1) priors
        self.alpha = np.ones(size)
        self.beta = np.ones(size)
        
        if use_heuristics:
            self.apply_heuristic_priors()

    def apply_heuristic_priors(self):
        # Prior strength equivalent to 10 pseudo-observations (virtual history weight)
        prior_strength = 10.0
        
        # Suspected Hotspots (prior expected mean = 0.15)
        hotspots = [(0, 0), (2, 2)]
        for r, c in hotspots:
            self.alpha[r, c] = 0.15 * prior_strength
            self.beta[r, c] = 0.85 * prior_strength
            
        # Suspected Oceans (prior expected mean = 0.95)
        oceans = [(0, 2), (2, 0)]
        for r, c in oceans:
            self.alpha[r, c] = 0.95 * prior_strength
            self.beta[r, c] = 0.05 * prior_strength
            
        # Suspected Standard Zones (prior expected mean = 0.60)
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if (r, c) not in hotspots and (r, c) not in oceans:
                    self.alpha[r, c] = 0.60 * prior_strength
                    self.beta[r, c] = 0.40 * prior_strength

    def sample_probabilities(self):
        # Draw a sample probability from Beta distribution for each grid cell
        sampled_map = np.zeros(self.size)
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                sampled_map[r, c] = np.random.beta(self.alpha[r, c], self.beta[r, c])
        return sampled_map

    def update_history(self, row, col, outcome):
        # Apply the discount factor to the historical success/failure counts
        # and bound them at 1.0 to prevent division by zero or parameter collapse
        if self.gamma < 1.0:
            self.alpha = np.maximum(self.alpha * self.gamma, 1.0)
            self.beta = np.maximum(self.beta * self.gamma, 1.0)
            
        if outcome == 1:
            self.alpha[row, col] += 1.0
        else:
            self.beta[row, col] += 1.0

    def get_estimated_probabilities(self):
        # Expected value of a Beta distribution is alpha / (alpha + beta)
        return self.alpha / (self.alpha + self.beta)


# --- SIMULATION EXECUTION ---
epochs = 1000
env = FederatedSatelliteEnv()

# Initialize our 3 experimental brokers
vanilla_broker = ThompsonSchedulingBroker(use_heuristics=False, gamma=1.0)
heuristic_static_broker = ThompsonSchedulingBroker(use_heuristics=True, gamma=1.0)
heuristic_adaptive_broker = ThompsonSchedulingBroker(use_heuristics=True, gamma=0.99)

# Metrics tracking
errors_vanilla = []
errors_static = []
errors_adaptive = []

print("Starting Satellite Scheduling Simulation...")
print("Grid Layout: Row 0, Col 0 = Middle East Hotspot | Row 0, Col 2 = Pacific Ocean")

for t in range(epochs):
    # 1. Update true environment probabilities (checks for concept drift)
    true_map = env.update_environment(t)
    
    # 2. Simulate scheduling actions: Schedulers sample and choose the top-2 cells to task
    for name, broker in [("vanilla", vanilla_broker), ("static", heuristic_static_broker), ("adaptive", heuristic_adaptive_broker)]:
        
        sampled_map = broker.sample_probabilities()
        
        # Flatten and pick the indices of the 2 highest sampled probabilities
        flat_indices = np.argsort(sampled_map.flatten())[-2:]
        selected_coords = [divmod(idx, 3) for idx in flat_indices]
        
        # 3. Execute observations and update the brokers with feedback
        for r, c in selected_coords:
            outcome = env.generate_feedback(r, c)
            broker.update_history(r, c, outcome)
            
    # 4. Record Mean Absolute Error (MAE) across all 9 zones
    errors_vanilla.append(np.mean(np.abs(vanilla_broker.get_estimated_probabilities() - true_map)))
    errors_static.append(np.mean(np.abs(heuristic_static_broker.get_estimated_probabilities() - true_map)))
    errors_adaptive.append(np.mean(np.abs(heuristic_adaptive_broker.get_estimated_probabilities() - true_map)))

    # Print out periodic status maps to inspect internal estimates
    if t in [0, 499, 999]:
        state_label = "INITIAL PRIOR STATE" if t == 0 else ("PRE-DRIFT STABLE STATE" if t == 499 else "POST-DRIFT RECOVERY STATE")
        print(f"\n--- {state_label} (Epoch {t}) ---")
        print("True Hidden Probabilities:\n", np.round(true_map, 2))
        print("Vanilla TS Estimates (No Heuristics, No Decays):\n", np.round(vanilla_broker.get_estimated_probabilities(), 2))
        print("Heuristic Static Estimates (Prior, No Decays):\n", np.round(heuristic_static_broker.get_estimated_probabilities(), 2))
        print("Heuristic Adaptive Estimates (Prior + 0.95 Decay):\n", np.round(heuristic_adaptive_broker.get_estimated_probabilities(), 2))

# Final Performance Summary
print("\n=== FINAL PERFORMANCE METRICS ===")
print(f"Vanilla TS         - Pre-Drift MAE (Epoch 499): {errors_vanilla[499]:.4f} | Post-Drift MAE (Epoch 999): {errors_vanilla[999]:.4f}")
print(f"Heuristic Static   - Pre-Drift MAE (Epoch 499): {errors_static[499]:.4f} | Post-Drift MAE (Epoch 999): {errors_static[999]:.4f}")
print(f"Heuristic Adaptive - Pre-Drift MAE (Epoch 499): {errors_adaptive[499]:.4f} | Post-Drift MAE (Epoch 999): {errors_adaptive[999]:.4f}")