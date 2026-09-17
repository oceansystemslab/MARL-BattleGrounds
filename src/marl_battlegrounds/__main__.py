"""Run package commands with ``python -m marl_battlegrounds --help``.

The command parser loads only the selected operation. Importing this module does
not run a command; module execution returns the command's documented exit code.
"""

from marl_battlegrounds._cli import main

if __name__ == "__main__":
    raise SystemExit(main())
