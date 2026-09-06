from __future__ import annotations

from tools.supersede_alphazuma_55_waiting_switch import _matching_switches


def test_matching_switches_requires_script_and_contract():
    inventory = [
        {"pid": 1, "command": "python switch.py --preregistration old.json"},
        {"pid": 2, "cmdline": "python switch.py --preregistration new.json"},
        {"pid": 3, "cmdline": "python other.py --preregistration old.json"},
    ]

    assert _matching_switches(
        inventory,
        script_name="switch.py",
        preregistration_name="old.json",
    ) == [inventory[0]]
