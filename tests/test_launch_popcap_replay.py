"""Tests for the read-only replay launcher."""

from tools.launch_popcap_replay import direct_launch_environment


def test_direct_launch_environment_disables_dpi_virtualization() -> None:
    environment = direct_launch_environment(3620)

    assert environment["SteamAppId"] == "3620"
    assert environment["SteamGameId"] == "3620"
    assert environment["__COMPAT_LAYER"] == "HIGHDPIAWARE"
