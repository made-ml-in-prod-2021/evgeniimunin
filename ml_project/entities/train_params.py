from dataclasses import dataclass, field

@dataclass()
class TrainingParams:
    model_type: str = field(default='RandomForestClassifier')
    max_depth: int = field(default=5)
    random_state: int = field(default=42)