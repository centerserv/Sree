import hashlib
import numpy as np
from abc import ABC  # or from layers.pattern import Validator if you keep that import

class PermanenceValidator(ABC):
    def __init__(self):
        # in-memory ledger
        self.ledger = {}

    def validate(self, preds, probs):
        V_b = []
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            data = f"{pred}{prob}".encode()
            hash_val = hashlib.sha256(data).hexdigest()
            # first-pass rows count as match, then only flag true mismatches :contentReference[oaicite:6]{index=6}
            if i not in self.ledger:
                V_b.append(1.0)
            elif self.ledger[i] == hash_val:
                V_b.append(1.0)
            else:
                V_b.append(0.0)
            self.ledger[i] = hash_val
        return np.array(V_b)

