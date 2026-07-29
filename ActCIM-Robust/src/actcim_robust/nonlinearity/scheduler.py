from __future__ import annotations


class CurriculumAlphaScheduler:
    def __init__(
        self,
        alpha_start: float = 0.1,
        alpha_end: float = 0.6,
        total_epochs: int = 50,
        power: float = 1.0,
        warmup_epochs: int = 0,
    ) -> None:
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_epochs = total_epochs
        self.power = power
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0
        self._current_alpha = alpha_start

    def get_alpha(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.alpha_start

        effective_epoch = epoch - self.warmup_epochs
        effective_total = self.total_epochs - self.warmup_epochs

        if effective_total <= 0:
            return self.alpha_end

        t = min(effective_epoch / effective_total, 1.0)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (t ** self.power)
        return alpha

    def step(self) -> float:
        self._current_alpha = self.get_alpha(self._current_epoch)
        self._current_epoch += 1
        return self._current_alpha

    @property
    def current_alpha(self) -> float:
        return self._current_alpha

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def reset(self) -> None:
        self._current_epoch = 0
        self._current_alpha = self.alpha_start


class SensitivityProbabilityCalculator:
    def __init__(
        self,
        sensitivity_scores: dict[str, float],
        p_min: float = 0.15,
        p_max: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        self._scores = dict(sensitivity_scores)
        self.p_min = p_min
        self.p_max = p_max
        self.gamma = gamma

        scores = list(self._scores.values())
        if len(scores) == 0:
            self._probs: dict[str, float] = {}
            return

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        self._probs = {}
        for name, score in self._scores.items():
            if score_range > 0:
                normalized = (score - min_score) / score_range
            else:
                normalized = 1.0
            p = self.p_min + (self.p_max - self.p_min) * (normalized ** self.gamma)
            self._probs[name] = p

    def get_probabilities(self) -> dict[str, float]:
        return dict(self._probs)

    def get_top_layers(self, fraction: float = 0.5) -> list[str]:
        if fraction <= 0 or fraction > 1.0:
            raise ValueError("fraction must be in (0, 1]")

        n = max(1, int(len(self._scores) * fraction))
        sorted_layers = sorted(self._scores, key=self._scores.get, reverse=True)
        return sorted_layers[:n]
