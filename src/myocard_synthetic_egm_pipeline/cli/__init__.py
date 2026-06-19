"""CLI entry points for the producer commands.

Each `*_cmd.py` module exposes a ``main()`` function wired in
``[project.scripts]`` as a console script:

- ``synthegm-generate-dataset CONFIG.yaml`` — Finitewave-driven N-sim
  clean dataset, optionally mixed against an iafdb-pipeline noise
  bank in the same run.
- ``synthegm-mix CONFIG.yaml`` — standalone mixer on an already-written
  clean ClassifierBank.

Both CLIs share the YAML loader + typed config builders in
:mod:`._config`.
"""
