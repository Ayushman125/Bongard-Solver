# Bongard-Problem-Solver

This project is a sophisticated system for solving Bongard problems, which are a classic challenge in visual reasoning. It combines a powerful System-1 (perceptual) component with a planned System-2 (reasoning) component.

## Project Structure

- **`.github/workflows/`**: Contains CI/CD pipelines for automated testing.
- **`integration/`**: Modules for integrating different parts of the system (e.g., schedulers, profilers).
- **`schemas/`**: JSON schemas for data validation.
- **`src/`**: Core source code for the Bongard solver.
  - `system1_al.py`: The System-1 Abstraction Layer for feature extraction.
  - `drift_monitor.py`: Monitors for data drift.
- **`tests/`**: Unit and integration tests.

## Versioning

Each core module in `src/` and `integration/` contains a `__version__` attribute.

---
*This project is being developed with the assistance of an AI programming partner.*
