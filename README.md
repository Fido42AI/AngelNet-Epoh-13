# AngelNet Epoh 13

AngelNet Epoh 13 is a research prototype that explores **Gravitational Tensor Cognition (GTC)** – a metaphor where intentions,
perception, and actions emerge from the dynamics of interacting semantic tensors. The code base contains building blocks for
simulating the cognitive field, sensors, regulators, and an MNIST-based demo training loop.

## Features

- Universal transformer that converts modality-specific inputs into a shared semantic field and can decode vectors back.
- Tensor-field based memory that stores class fields, resonance dynamics, and curvature snapshots.
- Modular architecture with intention, action, goal, and meta-reflection subsystems.
- MNIST demo that exercises the components while persisting state between runs.
- Baseline automated tests and linting integrated into CI.

## Repository Map

```
AngelNet-Epoh-13/
├── AngelNet-Epoh-13-Overview        # Conceptual description of the project
├── action_signal.py                 # Field-to-action transduction logic
├── angel_goal_module.py             # Goal tracking and reward shaping
├── angel_graph.py                   # Graph visualisation helpers
├── angelnet_core.py                 # Main neural network + tensor field orchestration
├── angel_sense.py                   # Field sensing and history persistence
├── cogni_core.py / cyber_core.py    # Cognitive regulation and cybernetic feedback
├── curvature_movie.py               # Curvature metrics over time
├── global_tensor_field.py           # Field aggregation utilities
├── intention_core.py                # Intention synthesis
├── main.py                          # MNIST training entry point
├── tensor_field_censor.py           # Field stability management and archives
├── tensor_global_vector_map.py      # Class fields, resonance, vector interpretation
├── universal_transformer.py         # Data-type specific vectorisation and decoding
├── tests/                           # Baseline smoke tests
├── docs/                            # Additional architecture documentation
└── reports/                         # Audit outputs and operational notes
```

A more detailed module-level overview is available in [`docs/architecture.md`](docs/architecture.md).

## Requirements

- Python 3.11+
- pip 23+
- CPU execution is supported; CUDA is optional.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index URL ensures that CPU wheels for PyTorch and TorchVision are available in constrained environments. Remove it
if you already have the packages installed.

## Environment Variables

Create a `.env` file based on [.env.example](.env.example) to customise persistent storage:

```
ANGELNET_STORAGE_DIR=.angelnet_state
```

If unset, AngelNet stores its archives under `~/.cache/angelnet_epoh13`.

## Running the Demo

```bash
source .venv/bin/activate
python main.py --epochs 1 --batch-size 32 --storage-dir .angelnet_state
```

The script downloads MNIST (if absent), trains for the requested number of epochs, persists tensor archives in the storage
directory, and saves metric visualisations per epoch (`metrics_epoch_<N>.png`).

## Linting and Tests

```bash
ruff check .
pytest
```

CI runs the same commands via `.github/workflows/ci.yml`.

> [!NOTE]
> The smoke tests automatically skip when PyTorch is not installed. Install the CPU wheels as shown in the installation
> section to execute them locally.

## Troubleshooting

- **Torch installation issues**: ensure you are using Python 3.11+ and, if necessary, supply the CPU wheel index as shown above.
- **Large persistent archives**: clean or relocate the storage directory defined by `ANGELNET_STORAGE_DIR` to reset the model
  between runs.

## License

AngelNet Epoh 13 is distributed under the terms of the [MIT License](LICENSE).
