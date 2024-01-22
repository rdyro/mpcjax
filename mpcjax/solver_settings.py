from dataclasses import dataclass

@dataclass(frozen=True)
class SolverSettings:
    solver: str = "sqp"
    linesearch: str = "scan"
    max_it: int = 10
    maxls: int = 200