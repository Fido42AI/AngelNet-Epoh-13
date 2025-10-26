# AngelNet Epoh 13 Architecture

AngelNet Epoh 13 implements the Gravitational Tensor Cognition (GTC) paradigm. The system is composed of cooperating modules
that exchange semantic tensors instead of tokens. This document summarises the most important components, their relationships,
and relevant data flows.

## High-Level Flow

```
Input → UniversalTransformer → AngelNet (fc1/fc2/fc3) → TensorGlobalVectorMap →
  ├─ GlobalTensorField ──▶ CyberCore / AngelSense / TensorFieldCensor
  ├─ IntentionCore ──▶ ActionSignalLayer
  ├─ MetaReflection ──▶ Reporting
  └─ AngelGoalModule ──▶ CogniCore (mood & reward)
```

1. **UniversalTransformer** converts modality-specific inputs (e.g., MNIST images) into a shared semantic vector space and can
   decode the latent vectors for inspection.
2. **AngelNet Core** applies feed-forward layers to the latent vectors, injects class fields from `TensorGlobalVectorMap`, and
   routes the result into downstream subsystems.
3. **TensorGlobalVectorMap** tracks class-specific vector fields, resonance factors, and provides interpretation utilities.
4. **CyberCore / AngelSense / TensorFieldCensor** monitor the dynamics of the tensor field, compress histories, and regulate
   instability.
5. **IntentionCore**, **ActionSignalLayer**, and **AngelGoalModule** derive intentions and actions based on field curvature and
   goal attainment.
6. **MetaReflection** and **AngelGraph** keep summaries useful for diagnostics and visualisation.

## Module Responsibilities

- `angelnet_core.AngelNet` orchestrates the entire pipeline, manages optimisers, and mediates between modules.
- `universal_transformer.UniversalTransformer` stores per-modality linear transforms/decoders with optional persistence.
- `tensor_global_vector_map.TensorGlobalVectorMap` persists class fields, interprets vectors, and handles clustering.
- `tensor_field_censor.TensorFieldCensor` acts as a gatekeeper, ensuring only meaningful field updates are archived.
- `angel_sense.AngelSense` records sensed field states with delta-compressed histories.
- `cogni_core.CogniCore` normalises reward signals into a mood measure used by the intention system.
- `angel_goal_module.AngelGoalModule` compares performance against a target accuracy, translating outcomes into rewards.
- `meta_reflection.MetaReflection` and `angel_graph.AngelGraph` expose long-term diagnostics for analysis and visualisation.

## Persistence Layout

AngelNet now stores archives in a configurable directory (`ANGELNET_STORAGE_DIR` or `~/.cache/angelnet_epoh13`):

```
<storage_dir>/
├── transformer_archive/        # UniversalTransformer segments
├── field_archive/              # TensorFieldCensor snapshots
├── class_fields.pth.gz         # TensorGlobalVectorMap class fields
├── vector_history.pth.gz       # TensorGlobalVectorMap vector history
├── sense_history.pth.gz        # AngelSense compressed history
└── ideal_fields.pth.gz         # TensorFieldCensor ideal fields
```

Tests override the storage location with a temporary directory to keep the repository clean.

## External Dependencies

- **PyTorch / TorchVision** provide tensor operations and the MNIST dataset.
- **Matplotlib** generates metric plots at the end of each epoch.

## Extension Points

- Additional data modalities can be introduced via `UniversalTransformer.add_data_type`.
- Custom goals or reward shaping strategies can extend `AngelGoalModule`.
- Alternative persistence strategies can be implemented by subclassing `TensorGlobalVectorMap` or `AngelSense`.

## Known Gaps

- Field visualisations (`AngelGraph`) currently target interactive usage and are not exercised in automated tests.
- The MNIST demo uses a simple fully connected backbone; experimenting with convolutional encoders would improve accuracy.
