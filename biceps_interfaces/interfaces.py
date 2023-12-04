import dataclasses
from typing import Any, Dict, List, Optional

import dataclasses_json


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Progress:
    """Descriptor of the trianing progress to inform the user."""

    epoch: int
    max_epochs: int
    step: int
    max_steps: int
    phase: str
    loss: float
    error: str = None
    done: bool = False


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Identity:
    """Descriptor of the identity of the client."""

    machine_id: str
    uid: str = ''
    devices: Optional[List[Dict[str, Any]]] = None
    remote_address: str = ''
    port: int = 0


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ComputeInfo:
    """Descriptor of the compute to be executed.

    The graph should be sent separately as a bytes file.
    """

    training: bool
    batch_size: int
    data_len: int
    use_mixed_precision: bool
    params: Dict[str, List[int]]
    num_epochs: Optional[int] = None
    stream_weights: Optional[bool] = None
    optimizers: Optional[str] = None
    lr_schedulers: Optional[str] = None
    seed: Optional[int] = None
    eval_freq: Optional[int] = None
