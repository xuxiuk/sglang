# CSD evaluation workflows

This directory contains reusable CSD calibration, launch, evaluation, merge,
plotting, and summarization tools. Runtime artifacts are written below
`runs/` and are intentionally not tracked by Git.

The shell entry points retain the model, environment, and data paths used for
the original experiments as overridable defaults. Set `ROOT`, `RUN_ROOT`,
model paths, calibration-table paths, and evaluator paths for the target
machine before running them. Each formal run should record the source commit,
command line, input-table digest, and dataset revision alongside its external
artifact bundle.

This branch does not retain evaluation results. Full model outputs, request
traces, server metadata, and large JSON files belong in external storage.
