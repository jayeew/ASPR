# Fig3

- `code/`: implementation modules (one physical copy).
- `data/`: experimental evidence, tables and rendering inputs.
- `final/`: accepted figure files.
- `scripts/`: figure-specific standalone commands.

Original paths remain symbolic references for saved evidence and existing imports, not duplicate datasets. Shared study assets remain in the shared experiment directories.

```bash
python3 -m figures fig3 paths
python3 -m figure_pipeline.innovation_five_dimensions --help
```

The five-dimension study guide lists the required local inputs and output paths.
[Five-dimension study](data/five_dimensions/README.md)
