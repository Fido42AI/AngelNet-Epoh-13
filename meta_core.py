# === meta_core.py ===
# AngelNet v13.4 – Metacognitive Supervisor
# Author: Fedorenko Bohdan
# Epoch XIII – April 2025

class MetaCore:
    def __init__(self, cogni, reflex, concepts, graph, global_field=None, cyber_core=None):
        self.cogni = cogni
        self.reflex = reflex
        self.concepts = concepts
        self.graph = graph
        self.global_field = global_field
        self.cyber_core = cyber_core
        self.meta_log = []

        self.thresholds = {
            "unstable_steps": 5,
            "flat_mood": 0.01,
            "concepts_min": 3
        }

    def evaluate(self):
        mood_dev = self._mood_fluctuation()
        unstable = self.reflex.detect_instability()
        concepts_count = len(self.concepts.concepts)

        # Cyber analysis if available
        if self.cyber_core:
            self.cyber_core.analyze()

        if unstable and concepts_count < self.thresholds["concepts_min"]:
            self._log("unstable_low_concepts", "Instability with weak conceptual core.")
        elif mood_dev < self.thresholds["flat_mood"]:
            self._log("flat_mood", "Mood stagnation – possible cognitive inertia.")
        elif concepts_count >= 6:
            self._log("growth_detected", "Cognitive growth detected.")
        else:
            self._log("neutral", "System is stable.")

        # Global field diagnostics
        if self.global_field:
            entropy = self.global_field.compute_entropy()
            self._log("field_entropy", f"Global field entropy: {entropy:.4f}")

    def _mood_fluctuation(self):
        moods = self.cogni.mood_trace
        if len(moods) < 2:
            return 0.0
        return max(moods) - min(moods)

    def _log(self, code, message):
        self.meta_log.append((code, message))
        print(f"[MetaCore] {message}")

    def summary(self):
        print("\n=== MetaCore Summary ===")
        for i, (code, msg) in enumerate(self.meta_log):
            print(f"  • Step {i+1}: {msg}")