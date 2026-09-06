"""Fail-closed source guidance for the Zuma's Revenge simulator.

CircleShootApp reconstructs Zuma Deluxe and is useful as an ancestor source
oracle, but its behavior is not automatically authoritative for Zuma's
Revenge.  This module pins that source snapshot, verifies selected behavior in
the frozen retail Revenge runtime, and checks that the corresponding Python
implementation follows the target runtime rather than the ancestor when they
disagree.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from typing import Any, Callable, Iterable, Mapping


SOURCE_GUIDANCE_SCHEMA = "zuma-rl.circleshoot-revenge-source-guidance"
SOURCE_GUIDANCE_VERSION = 3

EXPECTED_CIRCLESHOOT_REMOTE = "https://github.com/alula/CircleShootApp.git"
EXPECTED_CIRCLESHOOT_COMMIT = "165d0fd30d977da7ad5ee6efe128d8af2178713b"
EXPECTED_CIRCLESHOOT_TREE = "22f9cb1dac19cc491bb1ec582cff7d51348e3a11"
EXPECTED_CIRCLESHOOT_FILES = {
    "README.md": (
        4_022,
        "sha256:ae02396596cd40948d19aae367e28ebc83320c5f0a96e321ec384414eb5be9f7",
    ),
    "PopCap Framework License.txt": (
        2_131,
        "sha256:f5ce5040996ef3023ed0ed3dd931fc30bb9e0a329e799c7b8caafc1ba4a3f8f6",
    ),
    "Other Licenses.txt": (
        2_577,
        "sha256:56900ed4aa23906125a13c32040db9a7c528ef1e713adab2091bdbb410f37677",
    ),
    "source/CircleShoot/Board.cpp": (
        61_853,
        "sha256:6842bbbfa81a9647e9ad491963ee04503ea4f0c08bf374d9782579f2f4f83b3b",
    ),
    "source/CircleShoot/CurveMgr.cpp": (
        55_504,
        "sha256:e0be26938bd4ec75b3494b7c0c6c11c2f6982187498806bce92e9b77d97d3fd7",
    ),
    "source/CircleShoot/Ball.cpp": (
        19_438,
        "sha256:fac14fae447353dcf318547e34132473d4ffb987500f0f43d08ae0bbde8eb934",
    ),
    "source/CircleShoot/Bullet.cpp": (
        5_134,
        "sha256:47a61a7e5a22194fcf3fd29d3ead57e11d6c442920a20140c34025e9dfe16d1c",
    ),
    "source/CircleShoot/Gun.cpp": (
        13_046,
        "sha256:3b59340544dfd9fc96a6776ff37e242214bcd2f0dc26f976d406664a1f89483b",
    ),
    "source/CircleShoot/LevelParser.cpp": (
        31_782,
        "sha256:9728c847f24303452d70ae9a7d26eb10f6edccbf5a587dd55785074060ce8461",
    ),
    "source/CircleShoot/DataSync.cpp": (
        9_408,
        "sha256:6170c44117e7a18e96b4861c751366621739d6be0ca99af542e8ff636f915929",
    ),
    "source/CircleShoot/CurveData.cpp": (
        3_804,
        "sha256:a2b93c15b5e96b06f79437063126e810060736cede730e39e214a5222096cb8f",
    ),
    "source/CircleShoot/WayPoint.cpp": (
        12_480,
        "sha256:3dbf054abc1958a147af4945b1ccd2fc4e9deea51d92be8fb08e124d34bffa4e",
    ),
}

EXPECTED_RETAIL_RUNTIME_BYTES = 6_657_328
EXPECTED_RETAIL_RUNTIME_SHA256 = (
    "sha256:2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20"
)
EXPECTED_IMAGE_BASE = 0x00400000

GAP_SHOT_FUNCTION_VA = 0x0045C500
GAP_SHOT_FUNCTION_SIZE = 0x2E6
GAP_SHOT_FUNCTION_SHA256 = (
    "sha256:21f4236c5dfab2c4a99daa075af37bfbb535c18fe28b70135daaeddf452e9734"
)
GAP_SHOT_RADIUS_CONVERSION_CALL_VA = 0x0045C50F
GAP_SHOT_RADIUS_CONVERSION_TARGET_VA = 0x008D6250
GAP_SHOT_CALLER_VA = 0x004180C0
GAP_SHOT_DIRECT_CALL_VA = 0x004180CE

GAP_SHOT_INSTRUCTION_BLOCKS = {
    "setup": (
        0x0045C500,
        bytes.fromhex(
            "558bec83ec2c538b5d0cd943385657e83c9d47008b4d088b710c8b7e08"
            "8945ecdb45ec83c60403c085ffd95dec8945e8d945ecdcc8d95dec"
        ),
    ),
    "latched_point_compare": (
        0x0045C5BB,
        bytes.fromhex(
            "d945f8d945e4dcc8d9c1decadec1d945ecded9dfe0f6c4410f"
        ),
    ),
    "sample_point_compare": (
        0x0045C63B,
        bytes.fromhex(
            "d900d865f4d95df8d94004d865f0d95de4d945e4d945f8dcc8d9c1"
            "decadec1d945ecded9dfe0f6c4417416"
        ),
    ),
    "sample_loop_advance": (
        0x0045C666,
        bytes.fromhex("037de8035ddc3b7dfc7c99"),
    ),
    "board_call_site": (
        GAP_SHOT_CALLER_VA,
        bytes.fromhex(
            "8b939c0000008b45e88b0c025751e82d4404008b939c0000"
        ),
    ),
}

PENDING_COLOR_FUNCTION_VA = 0x00458B00
PENDING_COLOR_FUNCTION_SIZE = 0x305
PENDING_COLOR_FUNCTION_SHA256 = (
    "sha256:8e677d7649174aadb591f6c87df650e66f3bfb549b7a73437e148c7dcdc84892"
)
PENDING_COLOR_DIRECT_CALL_VAS = (0x00456373, 0x0045902C)
CURVE_MANAGER_VTABLE_VA = 0x0096A204
CURVE_MANAGER_RANDOM_MOD_SLOT = 0xA4
CURVE_MANAGER_RANDOM_MOD_VA = 0x004B4B70
GLOBAL_RNG_SHIM_VA = 0x00401530
GLOBAL_RNG_WRAPPER_VA = 0x00617490
BALL_RANDOMIZE_FRAME_VA = 0x004020D0

PENDING_COLOR_INSTRUCTION_BLOCKS = {
    "repeat_and_max_clump_guard": (
        0x00458CCC,
        bytes.fromhex(
            "e85f88faff99b964000000f7f93bd67f3a8b0dd81f9f008b410480b864"
            "10000000751b8b93780100008b809c0000008b8c906c0100008b51148b"
            "4228eb038b416c3945e87d058b45eceb3f"
        ),
    ),
    "forced_single_and_candidate_rejection": (
        0x00458D17,
        bytes.fromhex(
            "83ff0a7d246a0153e8acfcffff83f801751785ff740c6a0a53e89bfcff"
            "ff3bc77e078b45eceb188bff8b4b108b018b55e48b80a400000052ffd0"
            "3b45ec74ea"
        ),
    ),
    "candidate_rejection_loop": (
        0x00458D40,
        bytes.fromhex("8b4b108b018b55e48b80a400000052ffd03b45ec74ea"),
    ),
    "color_commit_then_visual_frame": (
        0x00458D56,
        bytes.fromhex("8b4df08941148b7df08b57148955e4e86693faff"),
    ),
    "random_mod_method": (
        CURVE_MANAGER_RANDOM_MOD_VA,
        bytes.fromhex("558bece8b8c9f4ff99f77d088bc25dc2"),
    ),
    "global_rng_shim": (
        GLOBAL_RNG_SHIM_VA,
        bytes.fromhex(
            "803d5823a3000075176a00c6055823a30001e830d54c0050e862ce4c00"
            "83c408a110c79f0080b85d05000000740e80b85e050000007505e950ce"
            "4c00e91f5f2100"
        ),
    ),
    "ball_randomize_frame": (
        BALL_RANDOMIZE_FRAME_VA,
        bytes.fromhex(
            "a130c49f008a808809000084c08b4f148d91df010000741a83f9037507"
            "bae5010000eb0e84c0740a83f9047505bae60100008b0c95a8bf9e0056"
            "8b31e81ff4ffff99f77e345e8997f4000000c3"
        ),
    ),
}

ADVANCE_BACKWARD_FUNCTION_VA = 0x0045A230
ADVANCE_BACKWARD_FUNCTION_SIZE = 0x22D
ADVANCE_BACKWARD_FUNCTION_SHA256 = (
    "sha256:42180161642de82f9e9c999b100181bc5b1d9b8686e4650de3b9d07a2d9c8e96"
)
ADVANCE_BACKWARD_DIRECT_CALL_VAS = (0x0045E176,)

FORWARD_SUCK_FUNCTION_VA = 0x0045A570
FORWARD_SUCK_FUNCTION_SIZE = 0x373
FORWARD_SUCK_FUNCTION_SHA256 = (
    "sha256:58692f9b8317034420b9e7d73ccc4da0a5976b0c3c076879976bd86c72be1930"
)
FORWARD_SUCK_DIRECT_CALL_VAS = (0x0045AE13,)

UPDATE_SUCKING_FUNCTION_VA = 0x0045A8F0
UPDATE_SUCKING_FUNCTION_SIZE = 0x531
UPDATE_SUCKING_FUNCTION_SHA256 = (
    "sha256:974f981fe4f8b1961874a1bfce5499e46178a444b02f37a290d28d8672b50da5"
)
UPDATE_SUCKING_DIRECT_CALL_VAS = (0x0045DE6C,)

UPDATE_SETS_FUNCTION_VA = 0x0045AE30
UPDATE_SETS_FUNCTION_SIZE = 0x2AE
UPDATE_SETS_FUNCTION_SHA256 = (
    "sha256:250840dd0a7ec62ad047f72675cd93335aa7f106eeb55c853cdb84d0f21e95ec"
)
UPDATE_SETS_DIRECT_CALL_VAS = (0x0045E189,)

ROLLBACK_INSTRUCTION_BLOCKS = {
    "backward_entry_and_reverse_seed": (
        0x0045A230,
        bytes.fromhex(
            "558bec83ec2053568b75088b461083b8b00100000057741a8b80b0010000"
            "8b108bc88b8290000000ffd084c00f84f2010000837e6400c686a0010000"
            "000f84e10100008b46608b40043b46608945f07505e88f49470083be8801"
            "000000d9eed95df8c645ff00744b8b46608b78043bf87505e86e4947003b"
            "7e607505e8644947008b4660d946188b4f088b7804d999e80000003bf875"
            "05e8494947003b7e607505e83f4947008b5708c782e400000001000000"
        ),
    ),
    "backward_move_and_countdown": (
        0x0045A2F0,
        bytes.fromhex(
            "8b7f088b9fe400000085db7e3bd987e80000008b46100fb68857030000d9"
            "5df8d9471c8b560cd865f85151d95df4d945f4d91c245752e855d40d0083"
            "c3ff899fe4000000c645ff01"
        ),
    ),
    "backward_contact_propagation": (
        0x0045A378,
        bytes.fromhex(
            "807dff008b58080f845effffff80bbb400000000742a8b5610d9431cd865"
            "f80fb6825703000050518b4e0cd95df4d945f4d91c245351e8cdd30d00e9"
            "2bffffffd9471cd94738e88dbe47008945f4da65f4d94338e87fbe470089"
            "45f4da65f4d95df4d9431cd945f4ded9dfe0f6c4057a42c683b400000001"
            "8b7e08c645ff01e8c4c8fbffd9431cd945f4d9c0deead9c9d95df8e840be"
            "47008b760c8bf8e886da0d00d945f48b7508d95b1c884304e9b8feffffc6"
            "45ff00e9affeffff"
        ),
    ),
    "backward_entrance_latch": (
        0x0045A434,
        bytes.fromhex(
            "807dff00741ab81400000039867c010000c686a0010000017d0689867c01"
            "0000"
        ),
    ),
    "suck_direction_and_speed": (
        0x0045A8F0,
        bytes.fromhex(
            "558bec83ec38538b5d088b431083b8b0010000005657741a8bc88b89b001"
            "00008b118b8290000000ffd084c00f84f60400008b4b608b118955e8eb06"
            "8d642400ddd88b7b608d735c3bf67405e8d44247008b45e83bc70f84cc04"
            "00003b46047505e8bf4247008b45e88b780880bfc200000000897dfc750d"
            "83bfe0000000000f8f9d0400008b87e00000008bc8c1f90385c0894df8db"
            "45f88945ecd84b18d95df4"
        ),
    ),
    "backward_suck_connected_motion": (
        0x0045A9E4,
        bytes.fromhex(
            "8b7f08d9471cc787e000000000000000d865f4c687c2000000018b4b100f"
            "b691570300008b430cd95df0d945f05251897df8d91c245750e860cd0d008b"
            "9fdc00000085db747080bb6b0100000075678bb33401000085f6744b80bb68"
            "0100000074096a00e87389faff8bf085f6743580bb6b01000000750d6a008b"
            "c3e87baefaff84c0741f8b4d088b51100fb682570300008b790c6a00505356"
            "32c0e8fbd00d008b7df8d9432cd99b40010000d94330d99b4401000080bf"
            "b4000000008b5d088b7dfc0f85f3feffff"
        ),
    ),
    "backward_suck_contact_and_seed": (
        0x0045ACEE,
        bytes.fromhex(
            "8b46143b47140f85eb000000d9471cd94738e84bb547008945ecda65ecd946"
            "38e83db547008945ecda65ecd95decd9461cd945ecd8d1dfe0ddd9f6c4050f"
            "8afffbffff8b4b100fb691570300008b430c5251d91c245650e836ca0d008b"
            "7b08e86ebffbff8b7d08c686b4000000018b75fc33db56899ee0000000c686"
            "c200000001e8ccd5ffff84c07408389eb8000000740c899eec000000899ef0"
            "0000008b4df83999e40000007539c781e40000001e000000db86ec000000dc"
            "0db0779900d95df8d90558779900d855f8dfe0f6c4017505d95df8eb02ddd8"
            "d945f8d999e8000000578bc1e888f6ffff"
        ),
    ),
    "forward_suck_scale_and_motion": (
        0x0045A5AC,
        bytes.fromhex(
            "8b86e00000008b55088bc8c1f90385c0894dfcdb45fc8975f88945ecd84a"
            "18d95df40f8edb0200008b5de03b5f047505e8344647008b7b088b4508d947"
            "1cd845f4c787e000000000000000c687c200000000"
        ),
    ),
    "set_start_and_rollback_arm": (
        0x0045AE30,
        bytes.fromhex(
            "558bec83ec28538b5d08c683a1010000008b436083c35c56578b38897dfceb"
            "038b7dfc3bdb8b73047405e8b63d47003bfe0f846e0200003b7b047505e8"
            "a43d47008b77088a86ba00000084c0740a8b4d08c681a10100000180bec000"
            "0000000f84c701000083bea400000000750433ffeb4c8bbea800000085ff8b"
            "9eac0000007505e85e3d47003b5f047505e8543d47008b86a40000003bf88b"
            "50048b1b8955ec7405e83d3d47003b5dec750433ffeb0d3b5f047505e82a3d"
            "47008b7b088b86a400000085c0745f8b8eac0000008b50048b9ea800000085"
            "db894df48b0a894de474043bd87405e8f93c47008b55e43955f4750433c0eb"
            "2f85db7505e8e43c47008b45f48b40043b43048945f47515e8d13c47008b45"
            "f43b43047508e8c43c47008b45f48b400885ff747080bfba00000000756785"
            "c0746380b8c000000000755a8b4f143b48147552c787e00000000a000000c6"
            "87c2000000018b96ec0000008b86f000000083c2018997ec0000008987f000"
            "0000"
        ),
    ),
    "set_entrance_stop_and_delete": (
        0x0045AFCE,
        bytes.fromhex(
            "8b47608b088d5f5c3bdb894ddc7405e8333c47008b55dc3955fc751bd9eeb8"
            "2800000039877c010000d99f900100007d0689877c01000056e8a5b8ffff8b"
            "75fc3b73047505e8fd3b47008b068bfb85ff8945fc7505e8ed3b47003b7704"
            "7505e8e33b47003b73040f8415feffff8b4e048b1689118b068b4e04568948"
            "04e80934470083c404834308ffe9f4fdffff"
        ),
    ),
    "set_two_tick_aging": (
        0x0045B05C,
        bytes.fromhex(
            "84c0744980beba00000000745880bebb00000000751c8b96d400000081e201"
            "00008079054a83cafe4275078386bc0000000183bebc000000147d0983be94"
            "000000037c21c686c000000001eb188b4d088b46148b5108838482e4000000"
            "018d8482e4000000"
        ),
    ),
    "curve_update_suck_call": (
        0x0045DE58,
        bytes.fromhex(
            "57e852b1ffff57e87cb4ffff8bdfe805ebffff57e87fcaffff"
        ),
    ),
    "curve_update_tail_order": (
        0x0045E16E,
        bytes.fromhex(
            "8bdfe8bbb2ffff57e8b5c0ffff8bc7e85ecfffff57e8b8d0ffff57e8a2cc"
            "ffff8bc7e8ebebffff"
        ),
    ),
}

IS_LOSING_FUNCTION_VA = 0x0045C7F0
IS_LOSING_FUNCTION_SIZE = 0xA6
IS_LOSING_FUNCTION_SHA256 = (
    "sha256:c9413833e4417d0284fefd4fad328a6626b84c40660a804c35c54933b60c1dff"
)
IS_LOSING_DIRECT_CALL_VAS = (0x0041B010,)
IS_WINNING_FUNCTION_VA = 0x0045C8A0
IS_WINNING_FUNCTION_SIZE = 0x1E
IS_WINNING_FUNCTION_SHA256 = (
    "sha256:1ba64a4991845cebfae96d10e38071e30e8abc1fce595f9975bd4a6b091cb25b"
)
BOARD_END_CONDITIONS_FUNCTION_VA = 0x0041A050
BOARD_END_CONDITIONS_DIRECT_CALL_VAS = (0x0041FBC4,)

TERMINAL_INSTRUCTION_BLOCKS = {
    "board_win_inline_loop": (
        0x0041A1D7,
        bytes.fromhex(
            "8b939c0000008bb26003000033ff33c085f67e3781c26c0100008b0a837964"
            "0075298379700075238b491083b9b00100000075178b8b9c00000083c00183"
            "c20483c7013b81600300007ccf3bfec745c4010000000f85bc0d0000"
        ),
    ),
    "loss_prerequisites": (
        0x0045C7F6,
        bytes.fromhex(
            "80bea10100000053570f8589000000837e64000f847f0000008b461080b857"
            "0300000075738b46608b58043bd88d7e5c7505e8e82347003b5f047505e8de"
            "2347008b4b08d9411c8b4e0cdd5df0e86885ffff8945fcdb45fcdc5df0dfe0"
            "f6c4417436837e5800753083be88010000007f27"
        ),
    ),
    "loss_front_segment_suction": (
        0x0045C867,
        bytes.fromhex(
            "e8444300008b0085c0741483b8e0000000007f136a01e88e6afaff85c075ec"
            "b0015f5b8be55dc3"
        ),
    ),
    "board_loss_loop": (
        0x0041AFED,
        bytes.fromhex(
            "33ff3bf7897dc8c745c4ffffffff7e3ac745cc6c0100008b839c0000008b4d"
            "cc8b3408e8db17040084c0751b8b939c0000008345cc048345c80183c7013b"
            "ba600300007cd2eb03897dc48b8b9c0000008b81600300003945c8740f8b45"
            "c45053e84ee1ffff"
        ),
    ),
}

POWERUP_ACTIVATE_BOMB_FUNCTION_VA = 0x00456C50
POWERUP_ACTIVATE_BOMB_FUNCTION_SIZE = 0x39A
POWERUP_ACTIVATE_BOMB_FUNCTION_SHA256 = (
    "sha256:c506a22e8c6a6824ebd3e2119b5d1abbb70b5a9c6ab113de9edaa6eefca27258"
)
POWERUP_ACTIVATE_BOMB_DIRECT_CALL_VAS = (0x00457183,)

POWERUP_CURVE_ACTIVATE_FUNCTION_VA = 0x00457150
POWERUP_CURVE_ACTIVATE_FUNCTION_SIZE = 0x6E
POWERUP_CURVE_ACTIVATE_FUNCTION_SHA256 = (
    "sha256:32a639054a0ff0440d9c06daa35daa57c8e16f319b488b94de62340a110944f2"
)
POWERUP_CURVE_ACTIVATE_DIRECT_CALL_VAS = (0x00418807,)

POWERUP_EXPLODE_BALL_FUNCTION_VA = 0x00457680
POWERUP_EXPLODE_BALL_FUNCTION_SIZE = 0x152
POWERUP_EXPLODE_BALL_FUNCTION_SHA256 = (
    "sha256:9d090c7bc7433c25740e785204ff1b0beafc9f296057f631c5bebf0b52841f4f"
)
POWERUP_EXPLODE_BALL_DIRECT_CALL_VAS = (
    0x00456A55,
    0x00456D0E,
    0x004570D7,
    0x0045727C,
    0x0045764D,
    0x004584C7,
    0x0045C0ED,
)

POWERUP_BOARD_ACTIVATE_FUNCTION_VA = 0x00418720
POWERUP_BOARD_ACTIVATE_FUNCTION_SIZE = 0x4EC
POWERUP_BOARD_ACTIVATE_FUNCTION_SHA256 = (
    "sha256:00d07a69517fac90b4b9f0428a8f11da110be00af9408bff803e77014574198d"
)
POWERUP_BOARD_ACTIVATE_DIRECT_CALL_VAS = (0x00457784,)

POWERUP_EFFECT_INSTRUCTION_BLOCKS = {
    "effective_type_priority": (
        0x00457151,
        bytes.fromhex(
            "8b811c01000083f80e751a83b9c4000000007e0b8b81c800000083f80e"
            "75068b8120010000"
        ),
    ),
    "curve_effect_dispatch": (
        0x00457176,
        bytes.fromhex(
            "85c0c680c81f9f000175095152e8c8faffff59c383f8037512837a6400"
            "7427c782880100002c01000059c383f801751681ba84010000e80300007d"
            "0ac782840100002003000059c3"
        ),
    ),
    "explode_guard_and_waypoint": (
        0x0045768D,
        bytes.fromhex(
            "80bbba00000000568bf157897424140f8527010000807d100074358b4608"
            "0504100000b9010000000148088b46080188001000008b809c000000e864"
            "cd05008b4e108b018b53148b80cc00000052ffd0d9431ce86beb47008986"
            "b4010000"
        ),
    ),
    "explode_effective_type_and_bookkeeping": (
        0x00457727,
        bytes.fromhex(
            "8b831c01000083f80e74048bc8eb1a83bbc4000000007e0b8b8bc8000000"
            "83f90e750b8b8b2001000083f90e747483f80e751a83bbc4000000007e0b"
            "8b83c800000083f80e75068b83200100008b74241483848624010000018b"
            "4e0853e8970ffcff8b831c01000083f80e8b56088b8ac80e000074048bd8"
            "eb1a83bbc4000000007e0b8b83c800000083f80e75e88b9b20010000898c"
            "9eb4000000c686bc01000001"
        ),
    ),
    "bomb_active_list_collision": (
        0x00456C94,
        bytes.fromhex(
            "8b730880beba00000000897424100f85800000008b450c6a38e82eecfaff"
            "84c074728b47088b88f80f00008b90fc0f00008996ec000000898ef00000"
            "008b5f088bb3940000008b4e0481c3900000008d442410505156e8810bfb"
            "ff8bcb8bf8e8b80bfbff8b442410897e048b57048b75086a016a00508bce"
            "893ae86d090000"
        ),
    ),
    "board_all_curves_dispatch": (
        0x004187DC,
        bytes.fromhex(
            "8b979c00000083ba6003000000c745ec000000007e35c745e86c0100008b"
            "879c0000008b55e88b14108bcbe844e903008b45ec8b8f9c0000008345e8"
            "0483c0013b81600300008945ec7cd2"
        ),
    ),
}

CIRCLESHOOT_METHOD_SHA256 = {
    "is_losing": "sha256:8c521fa6b5ff3d4fd9744fe4f45e66b4a9b9ed63dfbad378781d6930f8a8ba2e",
    "is_winning": "sha256:5954e80bce04f4fc59e4ca5ad28ed9856e0b2c3587615e5a0e819681361517dd",
    "update_playing": "sha256:f0f9339f5a6660f36c475337c53095eb8d5ff156350f90d3f2fff53c1c9904d4",
    "advance_backward_balls": "sha256:730f1f38a6cbfe867333e5b55e695c19f771ee035d5d0e03e454384f9690b9e2",
    "update_sucking_balls": "sha256:28071e1ca6e52b51b7227c6b4286b1dd8ae645a0ce36e87ebfe02055c89493d0",
    "update_sets": "sha256:8e0bf6635fc25c434e3b9b609d1716b79a2e13a86acfdaed0d65841573ebcb5f",
    "activate_power": "sha256:b3074b3b308cfd14ebb18b9742c5f4918f7676ad03f73eaf8cc10aa2876784ad",
    "check_set": "sha256:60c65abf27917e9c08dd6dc5dd425a0062baefdebd97540b400ccac32466893e",
    "start_clear_count": "sha256:6acef334eb3177bc05cc6521d4b715b80f66fadaa50c1491216f279241d5e3e4",
    "activate_bomb": "sha256:a13bceef12ad68bce39b47d5558a90e10bdf7f097c6a51f02837ab6ac2f2debe",
}


class SourceGuidanceError(RuntimeError):
    """A source snapshot, retail runtime, or implementation failed closed."""


def _fail(reason: str) -> None:
    raise SourceGuidanceError(reason)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_path(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise SourceGuidanceError("artifact_unreadable") from error
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class _PESection:
    virtual_address: int
    raw_size: int
    raw_offset: int
    executable: bool


@dataclass(frozen=True)
class _PELayout:
    image_base: int
    sections: tuple[_PESection, ...]


def _parse_pe32(payload: bytes) -> _PELayout:
    if len(payload) < 0x40 or payload[:2] != b"MZ":
        _fail("runtime_dos_header_invalid")
    pe_offset = struct.unpack_from("<I", payload, 0x3C)[0]
    if (
        pe_offset + 24 > len(payload)
        or payload[pe_offset : pe_offset + 4] != b"PE\0\0"
    ):
        _fail("runtime_pe_header_invalid")
    machine, section_count = struct.unpack_from("<HH", payload, pe_offset + 4)
    optional_size = struct.unpack_from("<H", payload, pe_offset + 20)[0]
    optional_offset = pe_offset + 24
    if (
        machine != 0x014C
        or section_count <= 0
        or optional_size < 0x60
        or optional_offset + optional_size > len(payload)
        or struct.unpack_from("<H", payload, optional_offset)[0] != 0x10B
    ):
        _fail("runtime_pe32_header_invalid")
    image_base = struct.unpack_from("<I", payload, optional_offset + 28)[0]
    section_table = optional_offset + optional_size
    sections: list[_PESection] = []
    for index in range(section_count):
        offset = section_table + index * 40
        if offset + 40 > len(payload):
            _fail("runtime_pe_section_table_invalid")
        _, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", payload, offset + 8
        )
        characteristics = struct.unpack_from("<I", payload, offset + 36)[0]
        if raw_size <= 0 or raw_offset + raw_size > len(payload):
            _fail("runtime_pe_section_bounds_invalid")
        sections.append(
            _PESection(
                virtual_address=virtual_address,
                raw_size=raw_size,
                raw_offset=raw_offset,
                executable=bool(characteristics & 0x20000000),
            )
        )
    return _PELayout(image_base=image_base, sections=tuple(sections))


def _read_va(
    payload: bytes,
    layout: _PELayout,
    virtual_address: int,
    size: int,
) -> bytes:
    if size < 0 or virtual_address < layout.image_base:
        _fail("runtime_virtual_address_invalid")
    relative = virtual_address - layout.image_base
    for section in layout.sections:
        local = relative - section.virtual_address
        if 0 <= local and local + size <= section.raw_size:
            offset = section.raw_offset + local
            return payload[offset : offset + size]
    _fail("runtime_virtual_address_unmapped")


def _rel32_target(
    payload: bytes,
    layout: _PELayout,
    call_address: int,
) -> tuple[bytes, int]:
    instruction = _read_va(payload, layout, call_address, 5)
    if instruction[0] != 0xE8:
        _fail("runtime_call_opcode_mismatch")
    displacement = struct.unpack_from("<i", instruction, 1)[0]
    return instruction, call_address + 5 + displacement


def _rel32_branch_target(
    payload: bytes,
    layout: _PELayout,
    instruction_address: int,
    *,
    opcode: int,
) -> tuple[bytes, int]:
    instruction = _read_va(payload, layout, instruction_address, 5)
    if instruction[0] != opcode:
        _fail("runtime_branch_opcode_mismatch")
    displacement = struct.unpack_from("<i", instruction, 1)[0]
    return instruction, instruction_address + 5 + displacement


def _direct_rel32_xrefs(
    payload: bytes,
    layout: _PELayout,
    target_address: int,
) -> tuple[int, ...]:
    hits: list[int] = []
    for section in layout.sections:
        if not section.executable:
            continue
        raw = payload[
            section.raw_offset : section.raw_offset + section.raw_size
        ]
        for local in range(max(0, len(raw) - 4)):
            if raw[local] != 0xE8:
                continue
            source = layout.image_base + section.virtual_address + local
            displacement = struct.unpack_from("<i", raw, local + 1)[0]
            if source + 5 + displacement == target_address:
                hits.append(source)
    return tuple(hits)


def verify_gap_shot_instruction_blocks(
    blocks: Mapping[str, bytes],
) -> Mapping[str, Any]:
    """Classify already-extracted retail blocks and reject any byte drift."""

    expected_names = tuple(GAP_SHOT_INSTRUCTION_BLOCKS)
    if tuple(blocks) != expected_names:
        _fail("runtime_gap_shot_block_set_mismatch")
    for name, (_, expected) in GAP_SHOT_INSTRUCTION_BLOCKS.items():
        if blocks[name] != expected:
            _fail(f"runtime_gap_shot_{name}_bytes_mismatch")
    return {
        "projectile_radius_field_offset": 0x38,
        "converted_radius_local_offset": -0x14,
        "sample_step_local_offset": -0x18,
        "sample_step_expression": "2 * converted_projectile_radius",
        "distance_threshold_expression": "converted_projectile_radius ** 2",
        "distance_threshold_applies_to": [
            "latched_curve_point",
            "candidate_curve_sample",
        ],
        "verdict": "radius_squared_threshold_with_diameter_sample_step",
    }


def verify_retail_gap_shot_static(payload: bytes) -> Mapping[str, Any]:
    """Verify the pinned retail Revenge ``CheckGapShot`` implementation."""

    layout = _parse_pe32(payload)
    if layout.image_base != EXPECTED_IMAGE_BASE:
        _fail("runtime_image_base_mismatch")
    function = _read_va(
        payload,
        layout,
        GAP_SHOT_FUNCTION_VA,
        GAP_SHOT_FUNCTION_SIZE,
    )
    if _sha256_bytes(function) != GAP_SHOT_FUNCTION_SHA256:
        _fail("runtime_gap_shot_function_hash_mismatch")
    blocks = {
        name: _read_va(payload, layout, address, len(expected))
        for name, (address, expected) in GAP_SHOT_INSTRUCTION_BLOCKS.items()
    }
    semantics = verify_gap_shot_instruction_blocks(blocks)
    conversion_instruction, conversion_target = _rel32_target(
        payload,
        layout,
        GAP_SHOT_RADIUS_CONVERSION_CALL_VA,
    )
    if conversion_target != GAP_SHOT_RADIUS_CONVERSION_TARGET_VA:
        _fail("runtime_gap_shot_radius_conversion_target_mismatch")
    xrefs = _direct_rel32_xrefs(payload, layout, GAP_SHOT_FUNCTION_VA)
    if xrefs != (GAP_SHOT_DIRECT_CALL_VA,):
        _fail("runtime_gap_shot_direct_xrefs_mismatch")
    return {
        "image_base": layout.image_base,
        "function_virtual_address": GAP_SHOT_FUNCTION_VA,
        "function_size": GAP_SHOT_FUNCTION_SIZE,
        "function_sha256": GAP_SHOT_FUNCTION_SHA256,
        "radius_conversion_call": {
            "instruction_virtual_address": GAP_SHOT_RADIUS_CONVERSION_CALL_VA,
            "instruction_hex": conversion_instruction.hex(),
            "target_virtual_address": conversion_target,
        },
        "direct_call_xrefs": list(xrefs),
        "instruction_blocks": {
            name: {
                "virtual_address": GAP_SHOT_INSTRUCTION_BLOCKS[name][0],
                "bytes": value.hex(),
                "sha256": _sha256_bytes(value),
            }
            for name, value in blocks.items()
        },
        "semantics": semantics,
    }


def verify_pending_color_instruction_blocks(
    blocks: Mapping[str, bytes],
) -> Mapping[str, Any]:
    """Classify the extracted retail pending-color blocks fail-closed."""

    expected_names = tuple(PENDING_COLOR_INSTRUCTION_BLOCKS)
    if tuple(blocks) != expected_names:
        _fail("runtime_pending_color_block_set_mismatch")
    for name, (_, expected) in PENDING_COLOR_INSTRUCTION_BLOCKS.items():
        if blocks[name] != expected:
            _fail(f"runtime_pending_color_{name}_bytes_mismatch")
    return {
        "repeat_roll": "global_mtrand_output % 100",
        "repeat_condition": (
            "repeat_roll <= repeat_chance and current_run < max_clump"
        ),
        "candidate_generation": "global_mtrand_output % active_color_count",
        "candidate_rejection": "repeat while candidate == previous_color",
        "visual_frame_draw_order": "after_color_selection_before_list_insert",
        "verdict": "max_clump_guarded_rejection_then_visual_frame",
    }


def verify_retail_pending_color_static(payload: bytes) -> Mapping[str, Any]:
    """Verify the pending-ball color and RNG order in the retail runtime."""

    layout = _parse_pe32(payload)
    if layout.image_base != EXPECTED_IMAGE_BASE:
        _fail("runtime_image_base_mismatch")
    function = _read_va(
        payload,
        layout,
        PENDING_COLOR_FUNCTION_VA,
        PENDING_COLOR_FUNCTION_SIZE,
    )
    if _sha256_bytes(function) != PENDING_COLOR_FUNCTION_SHA256:
        _fail("runtime_pending_color_function_hash_mismatch")
    blocks = {
        name: _read_va(payload, layout, address, len(expected))
        for name, (address, expected) in (
            PENDING_COLOR_INSTRUCTION_BLOCKS.items()
        )
    }
    semantics = verify_pending_color_instruction_blocks(blocks)

    xrefs = _direct_rel32_xrefs(
        payload,
        layout,
        PENDING_COLOR_FUNCTION_VA,
    )
    if xrefs != PENDING_COLOR_DIRECT_CALL_VAS:
        _fail("runtime_pending_color_direct_xrefs_mismatch")

    random_slot = _read_va(
        payload,
        layout,
        CURVE_MANAGER_VTABLE_VA + CURVE_MANAGER_RANDOM_MOD_SLOT,
        4,
    )
    random_method = struct.unpack("<I", random_slot)[0]
    if random_method != CURVE_MANAGER_RANDOM_MOD_VA:
        _fail("runtime_pending_color_random_vtable_slot_mismatch")

    repeat_call, repeat_target = _rel32_target(
        payload,
        layout,
        0x00458CCC,
    )
    random_method_call, random_method_target = _rel32_target(
        payload,
        layout,
        0x004B4B73,
    )
    frame_call, frame_target = _rel32_target(
        payload,
        layout,
        0x00458D65,
    )
    frame_rng_call, frame_rng_target = _rel32_target(
        payload,
        layout,
        0x0040210C,
    )
    for actual, expected, reason in (
        (repeat_target, GLOBAL_RNG_SHIM_VA, "repeat_rng_target"),
        (random_method_target, GLOBAL_RNG_SHIM_VA, "candidate_rng_target"),
        (frame_target, BALL_RANDOMIZE_FRAME_VA, "visual_frame_target"),
        (frame_rng_target, GLOBAL_RNG_SHIM_VA, "visual_frame_rng_target"),
    ):
        if actual != expected:
            _fail(f"runtime_pending_color_{reason}_mismatch")
    rng_jump, rng_target = _rel32_branch_target(
        payload,
        layout,
        0x0040156C,
        opcode=0xE9,
    )
    if rng_target != GLOBAL_RNG_WRAPPER_VA:
        _fail("runtime_pending_color_global_rng_target_mismatch")

    return {
        "function_virtual_address": PENDING_COLOR_FUNCTION_VA,
        "function_size": PENDING_COLOR_FUNCTION_SIZE,
        "function_sha256": PENDING_COLOR_FUNCTION_SHA256,
        "direct_call_xrefs": list(xrefs),
        "curve_manager_random_mod_vtable": {
            "vtable_virtual_address": CURVE_MANAGER_VTABLE_VA,
            "slot_offset": CURVE_MANAGER_RANDOM_MOD_SLOT,
            "target_virtual_address": random_method,
        },
        "rng_routes": {
            "repeat_roll": {
                "instruction_hex": repeat_call.hex(),
                "target_virtual_address": repeat_target,
            },
            "candidate_modulo": {
                "instruction_hex": random_method_call.hex(),
                "target_virtual_address": random_method_target,
            },
            "visual_frame": {
                "selection_call_hex": frame_call.hex(),
                "selection_target_virtual_address": frame_target,
                "rng_call_hex": frame_rng_call.hex(),
                "rng_target_virtual_address": frame_rng_target,
            },
            "global_rng_shim_jump": {
                "instruction_hex": rng_jump.hex(),
                "target_virtual_address": rng_target,
            },
        },
        "instruction_blocks": {
            name: {
                "virtual_address": PENDING_COLOR_INSTRUCTION_BLOCKS[name][0],
                "bytes": value.hex(),
                "sha256": _sha256_bytes(value),
            }
            for name, value in blocks.items()
        },
        "semantics": semantics,
    }


def verify_rollback_instruction_blocks(
    blocks: Mapping[str, bytes],
) -> Mapping[str, Any]:
    """Classify target rollback/update blocks and reject any byte drift."""

    expected_names = tuple(ROLLBACK_INSTRUCTION_BLOCKS)
    if tuple(blocks) != expected_names:
        _fail("runtime_rollback_block_set_mismatch")
    for name, (_, expected) in ROLLBACK_INSTRUCTION_BLOCKS.items():
        if blocks[name] != expected:
            _fail(f"runtime_rollback_{name}_bytes_mismatch")
    return {
        "suck_count_ramp": (
            "arithmetic_shift_right_3_then_multiply_curve_reverse_speed"
        ),
        "suck_direction_flag_offset": 0xC2,
        "normal_rollback_direction_flag": 1,
        "loss_suction_direction_flag": 0,
        "global_reverse_seed": {
            "ball": "skull_side_front_ball",
            "backwards_count": 1,
            "backwards_speed": "curve_reverse_speed",
        },
        "connected_backward_propagation": (
            "move_contacted_rear_balls_by_current_speed_then_snap_new_contact"
        ),
        "entrance_motion_latch": {
            "first_ball_moved_backwards": True,
            "first_ball_moved_backwards_offset": 0x1A0,
            "minimum_stop_ticks": 20,
            "stop_time_offset": 0x17C,
        },
        "gap_contact_seed": {
            "backwards_count": 30,
            "backwards_speed": "max(combo_count * 1.5, 0.5)",
        },
        "set_removal_seed": {
            "suck_count": 10,
            "suck_direction_flag": 1,
            "minimum_entrance_stop_ticks": 40,
        },
        "set_visual_lifetime": (
            "increment_every_second_ball_update_to_20_frames_or_native_state_3"
        ),
        "curve_update_order": [
            "update_sucking_balls",
            "advance_balls",
            "advance_backward_balls",
            "remove_balls_at_front",
            "remove_balls_at_end",
            "update_sets",
            "update_powerups",
        ],
        "verdict": "direction_aware_scaled_suction_and_contact_rollback",
    }


def verify_retail_rollback_static(payload: bytes) -> Mapping[str, Any]:
    """Verify the rollback chain and its update order in retail Revenge."""

    layout = _parse_pe32(payload)
    if layout.image_base != EXPECTED_IMAGE_BASE:
        _fail("runtime_image_base_mismatch")
    functions = {
        "advance_backward_balls": (
            ADVANCE_BACKWARD_FUNCTION_VA,
            ADVANCE_BACKWARD_FUNCTION_SIZE,
            ADVANCE_BACKWARD_FUNCTION_SHA256,
            ADVANCE_BACKWARD_DIRECT_CALL_VAS,
        ),
        "forward_suck_helper": (
            FORWARD_SUCK_FUNCTION_VA,
            FORWARD_SUCK_FUNCTION_SIZE,
            FORWARD_SUCK_FUNCTION_SHA256,
            FORWARD_SUCK_DIRECT_CALL_VAS,
        ),
        "update_sucking_balls": (
            UPDATE_SUCKING_FUNCTION_VA,
            UPDATE_SUCKING_FUNCTION_SIZE,
            UPDATE_SUCKING_FUNCTION_SHA256,
            UPDATE_SUCKING_DIRECT_CALL_VAS,
        ),
        "update_sets": (
            UPDATE_SETS_FUNCTION_VA,
            UPDATE_SETS_FUNCTION_SIZE,
            UPDATE_SETS_FUNCTION_SHA256,
            UPDATE_SETS_DIRECT_CALL_VAS,
        ),
    }
    function_proofs: dict[str, Mapping[str, Any]] = {}
    for name, (address, size, expected_digest, expected_xrefs) in (
        functions.items()
    ):
        function = _read_va(payload, layout, address, size)
        if _sha256_bytes(function) != expected_digest:
            _fail(f"runtime_rollback_{name}_function_hash_mismatch")
        xrefs = _direct_rel32_xrefs(payload, layout, address)
        if xrefs != expected_xrefs:
            _fail(f"runtime_rollback_{name}_direct_xrefs_mismatch")
        function_proofs[name] = {
            "virtual_address": address,
            "size": size,
            "sha256": expected_digest,
            "direct_call_xrefs": list(xrefs),
        }

    blocks = {
        name: _read_va(payload, layout, address, len(expected))
        for name, (address, expected) in ROLLBACK_INSTRUCTION_BLOCKS.items()
    }
    semantics = verify_rollback_instruction_blocks(blocks)

    call_routes = {
        "normal_or_loss_suction_dispatch": (0x0045DE6C, 0x0045A8F0),
        "forward_loss_suck_helper": (0x0045AE13, 0x0045A570),
        "advance_balls": (0x0045E170, 0x00459430),
        "advance_backward_balls": (0x0045E176, 0x0045A230),
        "remove_balls_at_front": (0x0045E17D, 0x0045B0E0),
        "remove_balls_at_end": (0x0045E183, 0x0045B240),
        "update_sets": (0x0045E189, 0x0045AE30),
        "update_powerups": (0x0045E190, 0x0045CD80),
    }
    route_proofs: dict[str, Mapping[str, Any]] = {}
    for name, (call_address, expected_target) in call_routes.items():
        instruction, target = _rel32_target(payload, layout, call_address)
        if target != expected_target:
            _fail(f"runtime_rollback_{name}_call_target_mismatch")
        route_proofs[name] = {
            "call_virtual_address": call_address,
            "instruction_hex": instruction.hex(),
            "target_virtual_address": target,
        }

    return {
        "functions": function_proofs,
        "call_routes": route_proofs,
        "instruction_blocks": {
            name: {
                "virtual_address": ROLLBACK_INSTRUCTION_BLOCKS[name][0],
                "bytes": value.hex(),
                "sha256": _sha256_bytes(value),
            }
            for name, value in blocks.items()
        },
        "semantics": semantics,
    }


def verify_powerup_effect_instruction_blocks(
    blocks: Mapping[str, bytes],
) -> Mapping[str, Any]:
    """Classify proven target power-up effects and reject byte drift."""

    expected_names = tuple(POWERUP_EFFECT_INSTRUCTION_BLOCKS)
    if tuple(blocks) != expected_names:
        _fail("runtime_powerup_effect_block_set_mismatch")
    for name, (_, expected) in POWERUP_EFFECT_INSTRUCTION_BLOCKS.items():
        if blocks[name] != expected:
            _fail(f"runtime_powerup_effect_{name}_bytes_mismatch")
    return {
        "effective_type_priority": [
            "primary",
            "live_previous",
            "secondary",
        ],
        "none_type_sentinel": 14,
        "per_explosion_waypoint": {
            "source_ball_field_offset": 0x1C,
            "destination_curve_field_offset": 0x1B4,
            "write_timing": "after_exploding_guard_before_powerup_dispatch",
        },
        "proximity_bomb": {
            "powerup_type": 0,
            "scope": "every_curve_then_each_active_ball",
            "skip_already_exploding": True,
            "physical_collision_pad": 56,
            "recursive_explode_ball": True,
        },
        "slow": {
            "powerup_type": 1,
            "replace_when_counter_below": 1_000,
            "replacement_ticks": 800,
        },
        "reverse": {
            "powerup_type": 3,
            "requires_nonempty_active_ball_list": True,
            "replacement_ticks": 300,
        },
        "trigger_bookkeeping": {
            "per_type_count_field_offset": 0x124,
            "per_type_cooldown_field_offset": 0xB4,
            "cooldown_source_board_field_offset": 0xEC8,
            "trigger_latch_field_offset": 0x1BC,
        },
        "verdict": (
            "per_explosion_waypoint_and_target_powerup_effects_recovered"
        ),
    }


def verify_retail_powerup_effects_static(payload: bytes) -> Mapping[str, Any]:
    """Verify bomb, slow, reverse, and explosion bookkeeping in Revenge."""

    layout = _parse_pe32(payload)
    if layout.image_base != EXPECTED_IMAGE_BASE:
        _fail("runtime_image_base_mismatch")
    functions = {
        "activate_bomb": (
            POWERUP_ACTIVATE_BOMB_FUNCTION_VA,
            POWERUP_ACTIVATE_BOMB_FUNCTION_SIZE,
            POWERUP_ACTIVATE_BOMB_FUNCTION_SHA256,
            POWERUP_ACTIVATE_BOMB_DIRECT_CALL_VAS,
        ),
        "curve_activate_power": (
            POWERUP_CURVE_ACTIVATE_FUNCTION_VA,
            POWERUP_CURVE_ACTIVATE_FUNCTION_SIZE,
            POWERUP_CURVE_ACTIVATE_FUNCTION_SHA256,
            POWERUP_CURVE_ACTIVATE_DIRECT_CALL_VAS,
        ),
        "curve_explode_ball": (
            POWERUP_EXPLODE_BALL_FUNCTION_VA,
            POWERUP_EXPLODE_BALL_FUNCTION_SIZE,
            POWERUP_EXPLODE_BALL_FUNCTION_SHA256,
            POWERUP_EXPLODE_BALL_DIRECT_CALL_VAS,
        ),
        "board_activate_power": (
            POWERUP_BOARD_ACTIVATE_FUNCTION_VA,
            POWERUP_BOARD_ACTIVATE_FUNCTION_SIZE,
            POWERUP_BOARD_ACTIVATE_FUNCTION_SHA256,
            POWERUP_BOARD_ACTIVATE_DIRECT_CALL_VAS,
        ),
    }
    function_proofs: dict[str, Mapping[str, Any]] = {}
    for name, (address, size, expected_digest, expected_xrefs) in (
        functions.items()
    ):
        function = _read_va(payload, layout, address, size)
        if _sha256_bytes(function) != expected_digest:
            _fail(f"runtime_powerup_effect_{name}_function_hash_mismatch")
        xrefs = _direct_rel32_xrefs(payload, layout, address)
        if xrefs != expected_xrefs:
            _fail(f"runtime_powerup_effect_{name}_direct_xrefs_mismatch")
        function_proofs[name] = {
            "virtual_address": address,
            "size": size,
            "sha256": expected_digest,
            "direct_call_xrefs": list(xrefs),
        }

    blocks = {
        name: _read_va(payload, layout, address, len(expected))
        for name, (address, expected) in (
            POWERUP_EFFECT_INSTRUCTION_BLOCKS.items()
        )
    }
    semantics = verify_powerup_effect_instruction_blocks(blocks)

    call_routes = {
        "curve_bomb_dispatch": (0x00457183, 0x00456C50),
        "board_curve_dispatch": (0x00418807, 0x00457150),
        "explode_board_dispatch": (0x00457784, 0x00418720),
        "bomb_recursive_explosion": (0x00456D0E, 0x00457680),
        "waypoint_float_to_int": (0x004576E0, 0x008D6250),
    }
    route_proofs: dict[str, Mapping[str, Any]] = {}
    for name, (call_address, expected_target) in call_routes.items():
        instruction, target = _rel32_target(payload, layout, call_address)
        if target != expected_target:
            _fail(f"runtime_powerup_effect_{name}_call_target_mismatch")
        route_proofs[name] = {
            "call_virtual_address": call_address,
            "instruction_hex": instruction.hex(),
            "target_virtual_address": target,
        }

    return {
        "functions": function_proofs,
        "call_routes": route_proofs,
        "instruction_blocks": {
            name: {
                "virtual_address": POWERUP_EFFECT_INSTRUCTION_BLOCKS[name][0],
                "bytes": value.hex(),
                "sha256": _sha256_bytes(value),
            }
            for name, value in blocks.items()
        },
        "semantics": semantics,
        "scope": {
            "spawn_scheduler_recovered": False,
            "fruit_collision_recovered": False,
            "reason": (
                "this proof is limited to ball-effect dispatch, active-chain "
                "bomb collision, and trigger bookkeeping"
            ),
        },
    }


def verify_terminal_instruction_blocks(
    blocks: Mapping[str, bytes],
) -> Mapping[str, Any]:
    """Classify natural win/loss predicates from exact target blocks."""

    expected_names = tuple(TERMINAL_INSTRUCTION_BLOCKS)
    if tuple(blocks) != expected_names:
        _fail("runtime_terminal_block_set_mismatch")
    for name, (_, expected) in TERMINAL_INSTRUCTION_BLOCKS.items():
        if blocks[name] != expected:
            _fail(f"runtime_terminal_{name}_bytes_mismatch")
    return {
        "curve_win_predicate": [
            "active_ball_list_empty",
            "pending_ball_list_empty",
            "special_or_boss_actor_absent",
        ],
        "board_win_quantifier": "all_curves_satisfy_curve_win_predicate",
        "curve_loss_predicate": [
            "no_active_explosion_sets",
            "active_ball_list_nonempty",
            "curve_dies_at_endpoint",
            "skull_side_ball_at_or_beyond_endpoint",
            "curve_bullet_list_empty",
            "global_reverse_counter_not_positive",
            "no_suck_counter_on_skull_connected_front_segment",
        ],
        "board_loss_quantifier": "any_curve_satisfies_curve_loss_predicate",
        "verdict": "curve_terminal_predicates_and_board_quantifiers_recovered",
    }


def verify_retail_terminal_static(payload: bytes) -> Mapping[str, Any]:
    """Verify natural terminal predicates and Board quantifiers in retail."""

    layout = _parse_pe32(payload)
    if layout.image_base != EXPECTED_IMAGE_BASE:
        _fail("runtime_image_base_mismatch")
    functions = {
        "is_losing": (
            IS_LOSING_FUNCTION_VA,
            IS_LOSING_FUNCTION_SIZE,
            IS_LOSING_FUNCTION_SHA256,
            IS_LOSING_DIRECT_CALL_VAS,
        ),
        "is_winning": (
            IS_WINNING_FUNCTION_VA,
            IS_WINNING_FUNCTION_SIZE,
            IS_WINNING_FUNCTION_SHA256,
            (),
        ),
    }
    function_proofs: dict[str, Mapping[str, Any]] = {}
    for name, (address, size, expected_digest, expected_xrefs) in (
        functions.items()
    ):
        function = _read_va(payload, layout, address, size)
        if _sha256_bytes(function) != expected_digest:
            _fail(f"runtime_terminal_{name}_function_hash_mismatch")
        xrefs = _direct_rel32_xrefs(payload, layout, address)
        if xrefs != expected_xrefs:
            _fail(f"runtime_terminal_{name}_direct_xrefs_mismatch")
        function_proofs[name] = {
            "virtual_address": address,
            "size": size,
            "sha256": expected_digest,
            "direct_call_xrefs": list(xrefs),
        }

    board_xrefs = _direct_rel32_xrefs(
        payload,
        layout,
        BOARD_END_CONDITIONS_FUNCTION_VA,
    )
    if board_xrefs != BOARD_END_CONDITIONS_DIRECT_CALL_VAS:
        _fail("runtime_terminal_board_direct_xrefs_mismatch")
    loss_call, loss_target = _rel32_target(payload, layout, 0x0041B010)
    if loss_target != IS_LOSING_FUNCTION_VA:
        _fail("runtime_terminal_board_loss_call_target_mismatch")
    set_losing_call, set_losing_target = _rel32_target(
        payload,
        layout,
        0x0041B04D,
    )
    if set_losing_target != 0x004191A0:
        _fail("runtime_terminal_set_losing_call_target_mismatch")

    blocks = {
        name: _read_va(payload, layout, address, len(expected))
        for name, (address, expected) in TERMINAL_INSTRUCTION_BLOCKS.items()
    }
    semantics = verify_terminal_instruction_blocks(blocks)
    return {
        "functions": function_proofs,
        "board_end_conditions": {
            "function_virtual_address": BOARD_END_CONDITIONS_FUNCTION_VA,
            "direct_call_xrefs": list(board_xrefs),
            "loss_predicate_call": {
                "instruction_virtual_address": 0x0041B010,
                "instruction_hex": loss_call.hex(),
                "target_virtual_address": loss_target,
            },
            "set_losing_call": {
                "instruction_virtual_address": 0x0041B04D,
                "instruction_hex": set_losing_call.hex(),
                "target_virtual_address": set_losing_target,
            },
            "win_predicate_form": "inlined",
        },
        "instruction_blocks": {
            name: {
                "virtual_address": TERMINAL_INSTRUCTION_BLOCKS[name][0],
                "bytes": value.hex(),
                "sha256": _sha256_bytes(value),
            }
            for name, value in blocks.items()
        },
        "semantics": semantics,
    }


def _git(source_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceGuidanceError("circleshoot_git_query_failed") from error
    return result.stdout.strip()


def verify_circleshoot_source(source_root: Path) -> Mapping[str, Any]:
    """Verify the detached, clean CircleShootApp ancestor snapshot."""

    source_root = source_root.resolve(strict=True)
    head = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD^{tree}")
    remote = _git(source_root, "remote", "get-url", "origin")
    branch = _git(source_root, "branch", "--show-current")
    dirty = _git(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if head != EXPECTED_CIRCLESHOOT_COMMIT:
        _fail("circleshoot_commit_mismatch")
    if tree != EXPECTED_CIRCLESHOOT_TREE:
        _fail("circleshoot_tree_mismatch")
    if remote != EXPECTED_CIRCLESHOOT_REMOTE:
        _fail("circleshoot_remote_mismatch")
    if branch:
        _fail("circleshoot_not_detached")
    if dirty:
        _fail("circleshoot_worktree_dirty")

    files: dict[str, Mapping[str, Any]] = {}
    for relative, (expected_size, expected_digest) in (
        EXPECTED_CIRCLESHOOT_FILES.items()
    ):
        path = source_root / relative
        try:
            size = path.stat().st_size
        except OSError as error:
            raise SourceGuidanceError(
                "circleshoot_reference_file_missing"
            ) from error
        digest = _sha256_path(path)
        if size != expected_size or digest != expected_digest:
            _fail("circleshoot_reference_file_mismatch")
        files[relative] = {"bytes": size, "sha256": digest}

    curve_mgr = source_root / "source/CircleShoot/CurveMgr.cpp"
    try:
        lines = curve_mgr.read_text(encoding="utf-8").splitlines(
            keepends=True
        )
    except OSError as error:
        raise SourceGuidanceError(
            "circleshoot_curve_manager_unreadable"
        ) from error
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("bool CurveMgr::CheckGapShot")
        )
        end = next(
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if line.startswith("int CurveMgr::GetRandomPendingBallColor")
        )
    except StopIteration as error:
        raise SourceGuidanceError(
            "circleshoot_gap_shot_method_missing"
        ) from error
    method = "".join(lines[start:end])
    required_fragments = (
        "int aBulDiameter = aBulRadius * 2;",
        "float aBulDiameterSq = (float)aBulDiameter * (float)aBulDiameter;",
        "for (int i = 1; i < aNumWayPoints; i += aBulDiameter)",
    )
    if any(fragment not in method for fragment in required_fragments):
        _fail("circleshoot_gap_shot_semantics_mismatch")
    if method.count("aBulDiameterSq >") != 2:
        _fail("circleshoot_gap_shot_threshold_count_mismatch")
    normalized_method = method.encode("utf-8")
    expected_method_digest = (
        "sha256:b9bc064ecbbca5fa4c54f27cf96e4b886066cde6de97d33d7f7f3338b5756002"
    )
    if _sha256_bytes(normalized_method) != expected_method_digest:
        _fail("circleshoot_gap_shot_method_hash_mismatch")

    try:
        pending_start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("void CurveMgr::AddPendingBall")
        )
        pending_end = next(
            index
            for index, line in enumerate(
                lines[pending_start + 1 :],
                pending_start + 1,
            )
            if line.startswith("void CurveMgr::AddBall")
        )
    except StopIteration as error:
        raise SourceGuidanceError(
            "circleshoot_pending_color_method_missing"
        ) from error
    pending_method = "".join(lines[pending_start:pending_end])
    pending_fragments = (
        "aBall->RandomizeFrame();",
        "if (Sexy::AppRand() % 100 <= mCurveDesc->mBallRepeat)",
        "} while (aNewColor == aPrevColor);",
        "aBall->SetType(aNewColor);",
    )
    if any(fragment not in pending_method for fragment in pending_fragments):
        _fail("circleshoot_pending_color_semantics_mismatch")
    if "mMaxClump" in pending_method:
        _fail("circleshoot_pending_color_unexpected_max_clump_guard")
    if not (
        pending_method.index("aBall->RandomizeFrame();")
        < pending_method.index("Sexy::AppRand() % 100")
        < pending_method.index("aBall->SetType(aNewColor);")
    ):
        _fail("circleshoot_pending_color_rng_order_mismatch")
    normalized_pending = pending_method.encode("utf-8")
    expected_pending_digest = (
        "sha256:f561d7088b63aab5aeb995a048f24241abf85224502fd60b9d1e8c16d9f8f600"
    )
    if _sha256_bytes(normalized_pending) != expected_pending_digest:
        _fail("circleshoot_pending_color_method_hash_mismatch")

    def extract_method(
        signature: str,
        next_signature: str,
        *,
        name: str,
    ) -> tuple[int, int, str]:
        try:
            method_start = next(
                index
                for index, line in enumerate(lines)
                if line.startswith(signature)
            )
            method_end = next(
                index
                for index, line in enumerate(
                    lines[method_start + 1 :],
                    method_start + 1,
                )
                if line.startswith(next_signature)
            )
        except StopIteration as error:
            raise SourceGuidanceError(
                f"circleshoot_{name}_method_missing"
            ) from error
        source = "".join(lines[method_start:method_end])
        if _sha256_bytes(source.encode("utf-8")) != (
            CIRCLESHOOT_METHOD_SHA256[name]
        ):
            _fail(f"circleshoot_{name}_method_hash_mismatch")
        return method_start + 1, method_end, source

    terminal_methods = {
        "is_losing": extract_method(
            "bool CurveMgr::IsLosing",
            "bool CurveMgr::IsWinning",
            name="is_losing",
        ),
        "is_winning": extract_method(
            "bool CurveMgr::IsWinning",
            "bool CurveMgr::CanRestart",
            name="is_winning",
        ),
    }
    losing_method = terminal_methods["is_losing"][2]
    for fragment in (
        "mHaveSets ||",
        "mBallList.empty() ||",
        "mWayPointMgr->GetEndPoint() > mBallList.back()->GetWayPoint()",
        "!mBulletList.empty() ||",
        "mBackwardCount > 0",
        "aBall->GetSuckCount() > 0",
        "aBall = aBall->GetPrevBall(true);",
    ):
        if fragment not in losing_method:
            _fail("circleshoot_is_losing_semantics_mismatch")
    winning_method = terminal_methods["is_winning"][2]
    if (
        "mBallList.empty() && mPendingBalls.empty()"
        not in winning_method
    ):
        _fail("circleshoot_is_winning_semantics_mismatch")

    rollback_methods = {
        "update_playing": extract_method(
            "void CurveMgr::UpdatePlaying",
            "void CurveMgr::UpdateLosing",
            name="update_playing",
        ),
        "advance_backward_balls": extract_method(
            "void CurveMgr::AdvanceBackwardBalls",
            "void CurveMgr::UpdateSuckingBalls",
            name="advance_backward_balls",
        ),
        "update_sucking_balls": extract_method(
            "void CurveMgr::UpdateSuckingBalls",
            "void CurveMgr::UpdateSets",
            name="update_sucking_balls",
        ),
        "update_sets": extract_method(
            "void CurveMgr::UpdateSets",
            "void CurveMgr::UpdatePowerUps",
            name="update_sets",
        ),
    }
    update_playing = rollback_methods["update_playing"][2]
    update_order_fragments = (
        "UpdateSuckingBalls();",
        "AdvanceBalls();",
        "AdvanceBackwardBalls();",
        "RemoveBallsAtFront();",
        "UpdateSets();",
        "UpdatePowerUps();",
    )
    if any(fragment not in update_playing for fragment in update_order_fragments):
        _fail("circleshoot_update_playing_semantics_mismatch")
    if list(map(update_playing.index, update_order_fragments)) != sorted(
        map(update_playing.index, update_order_fragments)
    ):
        _fail("circleshoot_update_playing_order_mismatch")

    advance_backward = rollback_methods["advance_backward_balls"][2]
    for fragment in (
        "SetBackwardsSpeed(1.0f);",
        "SetBackwardsCount(1);",
        "aBall->GetWayPoint() - aBackwardsSpeed",
        "aNextBall->GetWayPoint() - aBackwardsSpeed",
        "mFirstBallMovedBackwards = true;",
        "mStopTime = 20;",
    ):
        if fragment not in advance_backward:
            _fail("circleshoot_advance_backward_semantics_mismatch")
    update_sucking = rollback_methods["update_sucking_balls"][2]
    for fragment in (
        "float aSuck = aSuckCount / 8;",
        "aNextBall->GetWayPoint() - aSuck",
        "aNextBall->SetBackwardsCount(30);",
        "aBall->GetComboCount() * 1.5f",
        "aBackwardsSpeed = 0.5f;",
    ):
        if fragment not in update_sucking:
            _fail("circleshoot_update_sucking_semantics_mismatch")
    update_sets = rollback_methods["update_sets"][2]
    for fragment in (
        "if (aClearCount < 40)",
        "aNextBall->SetSuckCount(10);",
        "mAdvanceSpeed = 0.0f;",
        "mStopTime = 40;",
    ):
        if fragment not in update_sets:
            _fail("circleshoot_update_sets_semantics_mismatch")

    powerup_methods = {
        "activate_power": extract_method(
            "void CurveMgr::ActivatePower",
            "void CurveMgr::DrawCurve",
            name="activate_power",
        ),
        "check_set": extract_method(
            "bool CurveMgr::CheckSet",
            "void CurveMgr::DoScoring",
            name="check_set",
        ),
        "start_clear_count": extract_method(
            "void CurveMgr::StartClearCount",
            "void CurveMgr::ActivateBomb",
            name="start_clear_count",
        ),
        "activate_bomb": extract_method(
            "void CurveMgr::ActivateBomb",
            "void CurveMgr::ClearPendingSucks",
            name="activate_bomb",
        ),
    }
    activate_power = powerup_methods["activate_power"][2]
    for fragment in (
        "PowerType aPowerType = theBall->GetPowerTypeWussy();",
        "if (aPowerType == PowerType_Bomb)",
        "ActivateBomb(theBall);",
        "else if (aPowerType == PowerType_MoveBackwards)",
        "if (!mBallList.empty())",
        "mBackwardCount = 300;",
        "else if (aPowerType == PowerType_SlowDown)",
        "if (mSlowCount < 1000)",
        "mSlowCount = 800;",
    ):
        if fragment not in activate_power:
            _fail("circleshoot_activate_power_semantics_mismatch")
    check_set = powerup_methods["check_set"][2]
    set_traversal = (
        "Ball *anEndBall = aNextEnd->GetNextBall();",
        "Ball *aBall = aPrevEnd;",
        "while (aBall != anEndBall)",
        "StartClearCount(aBall);",
        "aBall = aBall->GetNextBall();",
    )
    if any(fragment not in check_set for fragment in set_traversal):
        _fail("circleshoot_check_set_powerup_traversal_mismatch")
    if list(map(check_set.index, set_traversal)) != sorted(
        map(check_set.index, set_traversal)
    ):
        _fail("circleshoot_check_set_powerup_order_mismatch")
    start_clear_count = powerup_methods["start_clear_count"][2]
    clear_fragments = (
        "if (theBall->GetClearCount() > 0)",
        "mLastClearedBallPoint = theBall->GetWayPoint();",
        "theBall->StartClearCount(",
        "if (theBall->GetPowerTypeWussy() != PowerType_None)",
        "mBoard->ActivatePower(theBall);",
    )
    if any(fragment not in start_clear_count for fragment in clear_fragments):
        _fail("circleshoot_start_clear_count_semantics_mismatch")
    if list(map(start_clear_count.index, clear_fragments)) != sorted(
        map(start_clear_count.index, clear_fragments)
    ):
        _fail("circleshoot_start_clear_count_order_mismatch")
    activate_bomb = powerup_methods["activate_bomb"][2]
    for fragment in (
        "for (BallList::iterator anItr = mBallList.begin();",
        "aBall->GetClearCount() == 0",
        "aBall->CollidesWithPhysically(theBall, 45)",
        "StartClearCount(aBall);",
    ):
        if fragment not in activate_bomb:
            _fail("circleshoot_activate_bomb_semantics_mismatch")

    return {
        "repository": EXPECTED_CIRCLESHOOT_REMOTE,
        "commit": head,
        "tree": tree,
        "detached_head": True,
        "clean_worktree": True,
        "files": files,
        "gap_shot": {
            "path": "source/CircleShoot/CurveMgr.cpp",
            "start_line": start + 1,
            "end_line": end,
            "normalized_source_sha256": expected_method_digest,
            "sample_step_expression": "2 * projectile_radius",
            "distance_threshold_expression": "(2 * projectile_radius) ** 2",
            "verdict": "diameter_squared_threshold_with_diameter_sample_step",
        },
        "pending_color": {
            "path": "source/CircleShoot/CurveMgr.cpp",
            "start_line": pending_start + 1,
            "end_line": pending_end,
            "normalized_source_sha256": expected_pending_digest,
            "repeat_condition": "repeat_roll <= repeat_chance",
            "max_clump_guard": False,
            "candidate_rejection": "repeat while candidate == previous_color",
            "visual_frame_draw_order": "before_color_selection",
            "verdict": "deluxe_rng_order_without_revenge_max_clump_guard",
        },
        "rollback": {
            "path": "source/CircleShoot/CurveMgr.cpp",
            "methods": {
                name: {
                    "start_line": method_start,
                    "end_line": method_end,
                    "normalized_source_sha256": (
                        CIRCLESHOOT_METHOD_SHA256[name]
                    ),
                }
                for name, (method_start, method_end, _) in (
                    rollback_methods.items()
                )
            },
            "suck_count_ramp": "integer_divide_by_8_without_curve_scale",
            "suck_direction_flag": False,
            "global_reverse_speed": "hardcoded_1.0",
            "gap_contact_seed": {
                "backwards_count": 30,
                "backwards_speed": "max(combo_count * 1.5, 0.5)",
            },
            "set_removal_seed": {
                "suck_count": 10,
                "minimum_entrance_stop_ticks": 40,
            },
            "curve_update_order": list(update_order_fragments),
            "verdict": "shared_topology_without_revenge_direction_or_scale",
        },
        "powerup_effects": {
            "path": "source/CircleShoot/CurveMgr.cpp",
            "methods": {
                name: {
                    "start_line": method_start,
                    "end_line": method_end,
                    "normalized_source_sha256": (
                        CIRCLESHOOT_METHOD_SHA256[name]
                    ),
                }
                for name, (method_start, method_end, _) in (
                    powerup_methods.items()
                )
            },
            "effective_type_source": "GetPowerTypeWussy",
            "match_clear_traversal": "previous_end_through_next_end",
            "last_cleared_waypoint_timing": (
                "every_new_clear_before_powerup_dispatch"
            ),
            "proximity_bomb": {
                "scope": "active_ball_list",
                "skip_already_clearing": True,
                "physical_collision_pad": 45,
                "recursive_clear": True,
            },
            "slow": {
                "replace_when_counter_below": 1_000,
                "replacement_ticks": 800,
            },
            "reverse": {
                "requires_nonempty_active_ball_list": True,
                "replacement_ticks": 300,
            },
            "verdict": (
                "shared_effect_topology_with_deluxe_bomb_pad_45"
            ),
        },
        "terminal": {
            "path": "source/CircleShoot/CurveMgr.cpp",
            "methods": {
                name: {
                    "start_line": method_start,
                    "end_line": method_end,
                    "normalized_source_sha256": (
                        CIRCLESHOOT_METHOD_SHA256[name]
                    ),
                }
                for name, (method_start, method_end, _) in (
                    terminal_methods.items()
                )
            },
            "curve_win_predicate": [
                "active_ball_list_empty",
                "pending_ball_list_empty",
            ],
            "curve_loss_predicate": [
                "no_active_explosion_sets",
                "active_ball_list_nonempty",
                "skull_side_ball_at_or_beyond_endpoint",
                "curve_bullet_list_empty",
                "global_reverse_counter_not_positive",
                "no_suck_counter_on_skull_connected_front_segment",
            ],
            "verdict": "ancestor_predicates_without_revenge_special_actor_gate",
        },
        "licensing_scope": {
            "game_source": "provided_as_is; modifications_identified_as_mit",
            "framework": "separate_popcap_framework_license",
            "assets": "not_included",
            "use_in_this_project": "internal_reference_only_no_redistribution",
        },
    }


def _find_function(tree: ast.AST, function_name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(matches) != 1:
        _fail("python_gap_function_count_mismatch")
    return matches[0]


def _assignment(function: ast.FunctionDef, name: str) -> ast.expr:
    matches: list[ast.expr] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(node.value)
    if len(matches) != 1:
        _fail(f"python_{name}_assignment_mismatch")
    return matches[0]


def _is_name(value: ast.AST, name: str) -> bool:
    return isinstance(value, ast.Name) and value.id == name


def _is_attribute(value: ast.AST, owner: str, attribute: str) -> bool:
    return (
        isinstance(value, ast.Attribute)
        and value.attr == attribute
        and _is_name(value.value, owner)
    )


def _is_times_two(value: ast.AST, operand: Callable[[ast.AST], bool]) -> bool:
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Mult)
        and operand(value.left)
        and isinstance(value.right, ast.Constant)
        and value.right.value == 2
    )


def _radius_threshold_comparisons(function: ast.FunctionDef) -> set[str]:
    operators: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        values = (node.left, *node.comparators)
        if not any(_is_name(value, "radius_squared") for value in values):
            continue
        operators.add(type(node.ops[0]).__name__)
    return operators


def verify_python_gap_contract(
    path: Path,
    *,
    function_name: str,
    implementation: str,
) -> Mapping[str, Any]:
    """Verify one Python gap-shot implementation through its normalized AST."""

    path = path.resolve(strict=True)
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise SourceGuidanceError("python_gap_source_unreadable") from error
    function = _find_function(tree, function_name)
    step = _assignment(function, "step")
    radius_squared = _assignment(function, "radius_squared")

    if implementation == "simulator":
        if not _is_times_two(
            step,
            lambda value: _is_attribute(value, "projectile", "radius"),
        ):
            _fail("python_simulator_gap_step_mismatch")
        radius_ok = (
            isinstance(radius_squared, ast.Call)
            and _is_name(radius_squared.func, "_f32_mul")
            and len(radius_squared.args) == 2
            and all(_is_name(value, "radius") for value in radius_squared.args)
            and not radius_squared.keywords
        )
    elif implementation == "mechanism_audit":
        if not _is_times_two(
            step,
            lambda value: _is_name(value, "projectile_radius"),
        ):
            _fail("python_audit_gap_step_mismatch")
        radius_ok = (
            isinstance(radius_squared, ast.Call)
            and _is_attribute(radius_squared.func, "np", "multiply")
            and len(radius_squared.args) == 2
            and all(
                isinstance(value, ast.Call)
                and _is_attribute(value.func, "np", "float32")
                and len(value.args) == 1
                and _is_name(value.args[0], "projectile_radius")
                for value in radius_squared.args
            )
            and len(radius_squared.keywords) == 1
            and radius_squared.keywords[0].arg == "dtype"
            and _is_attribute(
                radius_squared.keywords[0].value,
                "np",
                "float32",
            )
        )
    else:
        raise ValueError("unsupported gap implementation")
    if not radius_ok:
        _fail(f"python_{implementation}_gap_threshold_mismatch")
    operators = _radius_threshold_comparisons(function)
    if operators != {"Lt", "GtE"}:
        _fail(f"python_{implementation}_gap_comparison_mismatch")

    segment = ast.get_source_segment(text, function)
    if segment is None:
        _fail("python_gap_source_segment_missing")
    return {
        "implementation": implementation,
        "path": str(path),
        "function": function_name,
        "start_line": function.lineno,
        "end_line": function.end_lineno,
        "normalized_ast_sha256": _sha256_bytes(
            ast.dump(function, include_attributes=False).encode("utf-8")
        ),
        "source_segment_sha256": _sha256_bytes(segment.encode("utf-8")),
        "sample_step_expression": "2 * projectile_radius",
        "distance_threshold_expression": "projectile_radius ** 2",
        "comparison_operators": sorted(operators),
        "verdict": "matches_retail_revenge",
    }


def _attribute_chain(value: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _is_call_chain(value: ast.AST, chain: tuple[str, ...]) -> bool:
    return isinstance(value, ast.Call) and _attribute_chain(value.func) == chain


def _contains_named_compare(
    value: ast.AST,
    *,
    left: str,
    operator: type[ast.cmpop],
    right: str,
) -> bool:
    return any(
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], operator)
        and _is_name(node.left, left)
        and len(node.comparators) == 1
        and _is_name(node.comparators[0], right)
        for node in ast.walk(value)
    )


def verify_python_pending_color_contract(path: Path) -> Mapping[str, Any]:
    """Verify the simulator's target-specific pending-color RNG order."""

    path = path.resolve(strict=True)
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise SourceGuidanceError("python_pending_source_unreadable") from error
    function = _find_function(tree, "_append_pending_color")

    repeat_candidates = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and _contains_named_compare(
            node.test,
            left="roll",
            operator=ast.LtE,
            right="repeat",
        )
        and _contains_named_compare(
            node.test,
            left="current_run",
            operator=ast.Lt,
            right="max_clump",
        )
    ]
    if len(repeat_candidates) != 1:
        _fail("python_pending_repeat_max_clump_guard_mismatch")
    repeat_branch = repeat_candidates[0]

    rejection_loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and _is_name(node.test.left, "color")
        and len(node.test.comparators) == 1
        and _is_name(node.test.comparators[0], "previous")
    ]
    if len(rejection_loops) != 1:
        _fail("python_pending_candidate_rejection_loop_mismatch")
    rejection = rejection_loops[0]
    candidate_assignments = [
        node
        for node in ast.walk(rejection)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _is_name(node.targets[0], "color")
        and _is_call_chain(node.value, ("self", "_rand_mod"))
        and len(node.value.args) == 1
        and _is_attribute(node.value.args[0], "self", "active_num_colors")
    ]
    if len(candidate_assignments) != 1:
        _fail("python_pending_candidate_draw_mismatch")

    frame_draws = [
        node
        for node in ast.walk(function)
        if _is_call_chain(node, ("self", "rng", "next_u31"))
    ]
    pending_appends = [
        node
        for node in ast.walk(function)
        if _is_call_chain(node, ("self", "pending_colors", "append"))
    ]
    if len(frame_draws) != 1 or len(pending_appends) != 1:
        _fail("python_pending_visual_frame_or_insert_call_mismatch")
    frame_draw = frame_draws[0]
    pending_append = pending_appends[0]
    if not (
        repeat_branch.lineno
        < rejection.lineno
        < frame_draw.lineno
        < pending_append.lineno
    ):
        _fail("python_pending_rng_order_mismatch")

    segment = ast.get_source_segment(text, function)
    if segment is None:
        _fail("python_pending_source_segment_missing")
    return {
        "implementation": "simulator_pending_color",
        "path": str(path),
        "function": function.name,
        "start_line": function.lineno,
        "end_line": function.end_lineno,
        "normalized_ast_sha256": _sha256_bytes(
            ast.dump(function, include_attributes=False).encode("utf-8")
        ),
        "source_segment_sha256": _sha256_bytes(segment.encode("utf-8")),
        "repeat_condition": (
            "repeat_roll <= repeat_chance and current_run < max_clump"
        ),
        "candidate_rejection": "repeat while candidate == previous_color",
        "visual_frame_draw_order": "after_color_selection_before_list_insert",
        "verdict": "matches_retail_revenge",
    }


def _verify_normalized_ast_fragments(
    function: ast.FunctionDef,
    fragments: Iterable[str],
    *,
    reason: str,
) -> str:
    normalized = ast.unparse(function)
    if any(fragment not in normalized for fragment in fragments):
        _fail(reason)
    return normalized


def _python_function_receipt(
    text: str,
    function: ast.FunctionDef,
) -> Mapping[str, Any]:
    segment = ast.get_source_segment(text, function)
    if segment is None:
        _fail("python_source_segment_missing")
    return {
        "function": function.name,
        "start_line": function.lineno,
        "end_line": function.end_lineno,
        "normalized_ast_sha256": _sha256_bytes(
            ast.dump(function, include_attributes=False).encode("utf-8")
        ),
        "source_segment_sha256": _sha256_bytes(segment.encode("utf-8")),
    }


def verify_python_rollback_contract(path: Path) -> Mapping[str, Any]:
    """Verify the simulator's target-specific rollback core and order."""

    path = path.resolve(strict=True)
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise SourceGuidanceError("python_rollback_source_unreadable") from error

    functions = {
        name: _find_function(tree, name)
        for name in (
            "_update_sucking_balls",
            "_advance_backward_balls",
            "_remove_ball_at",
            "_update_sets",
            "_tick_curve",
        )
    }
    required = {
        "_update_sucking_balls": (
            "anchor.suck_count <= 0 or not anchor.suck_back",
            "speed = _f32_mul(np.float32(old_suck_count >> 3), "
            "np.float32(self.config.reverse_speed))",
            "moved.waypoint = _f32_sub(moved.waypoint, speed)",
            "anchor.suck_count = old_suck_count + 1",
            "previous.color != anchor.color",
            "front.backwards_count = 30",
            "backwards_speed = _f32_mul(np.float32(anchor.combo_count), "
            "np.float32(1.5))",
            "backwards_speed = _F32_HALF",
            "self._clear_pending_sucks(moved_last)",
        ),
        "_advance_backward_balls": (
            "self.first_ball_moved_backwards = False",
            "front.backwards_speed = np.float32(self.config.reverse_speed)",
            "front.backwards_count = 1",
            "for index in range(len(self.balls) - 1, -1, -1):",
            "ball.waypoint = _f32_sub(ball.waypoint, speed)",
            "ball.backwards_count -= 1",
            "rear.waypoint = _f32_sub(rear.waypoint, speed)",
            "speed = _f32_sub(rear.waypoint, target)",
            "rear.contact_next = True",
            "self.first_ball_moved_backwards = True",
            "self.stop_time = max(self.stop_time, 20)",
        ),
        "_remove_ball_at": (
            "next_ball.color == previous.color",
            "next_ball.suck_count = 10",
            "next_ball.suck_back = True",
            "next_ball.combo_count = ball.combo_count + 1",
            "self.advance_speed = np.float32(0.0)",
            "self.stop_time = max(self.stop_time, 40)",
        ),
        "_update_sets": (
            "self._have_sets = False",
            "if ball.exploding:",
            "self._have_sets = True",
            "if ball.should_remove:",
            "self._remove_ball_at(index)",
            "if ball.update_count % 2 == 0:",
            "ball.explode_frame += 1",
            "if ball.explode_frame >= 20:",
            "ball.should_remove = True",
        ),
    }
    for name, fragments in required.items():
        _verify_normalized_ast_fragments(
            functions[name],
            fragments,
            reason=f"python_rollback_{name}_contract_mismatch",
        )

    tick_curve = functions["_tick_curve"]
    calls = sorted(
        (
            node.lineno,
            node.col_offset,
            node.func.attr,
        )
        for node in ast.walk(tick_curve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _is_name(node.func.value, "self")
    )
    call_names = [name for _, _, name in calls]
    expected_order = [
        "_update_sucking_balls",
        "_advance_balls",
        "_advance_backward_balls",
        "_remove_front_for_rollout",
        "_remove_balls_at_end",
        "_update_sets",
        "_maybe_spawn_powerup",
    ]
    try:
        positions = [call_names.index(name) for name in expected_order]
    except ValueError as error:
        raise SourceGuidanceError(
            "python_rollback_update_call_missing"
        ) from error
    if positions != sorted(positions):
        _fail("python_rollback_update_order_mismatch")

    return {
        "implementation": "simulator_rollback_core",
        "path": str(path),
        "functions": {
            name: _python_function_receipt(text, function)
            for name, function in functions.items()
        },
        "suck_count_ramp": (
            "arithmetic_shift_right_3_then_multiply_reverse_speed"
        ),
        "normal_suck_direction": "explicit_backward",
        "global_reverse_seed_speed": "reverse_speed",
        "gap_contact_seed": "30_ticks_max_combo_times_1.5_floor_0.5",
        "set_removal_seed": "10_ticks_explicit_backward",
        "curve_update_order": expected_order,
        "verdict": "matches_retail_revenge_rollback_core",
    }


def verify_python_powerup_effect_contract(path: Path) -> Mapping[str, Any]:
    """Verify the simulator's statically recovered target power-up core."""

    path = path.resolve(strict=True)
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise SourceGuidanceError("python_powerup_source_unreadable") from error

    functions = {
        name: _find_function(tree, name)
        for name in (
            "_effective_ball_powerup",
            "_trigger_ball_powerup",
            "_begin_ball_explosion",
        )
    }
    required = {
        "_effective_ball_powerup": (
            "if ball.powerup_primary_type != none_type:",
            "return ball.powerup_primary_type",
            "ball.powerup_previous_ticks > 0",
            "ball.powerup_previous_type != none_type",
            "return ball.powerup_previous_type",
            "return ball.powerup_secondary_type",
        ),
        "_trigger_ball_powerup": (
            "self.active_powerup_color_counts[ball.color] = max(0, "
            "active_count - 1)",
            "self.powerup_field_124_by_type[powerup_type] += 1",
            "for candidate in self.balls:",
            "if candidate.exploding:",
            "self.config.proximity_bomb_collision_pad",
            "self._begin_ball_explosion(candidate)",
            "if self.slow_count < "
            "self.config.slow_powerup_replace_threshold:",
            "self.slow_count = self.config.slow_powerup_ticks",
            "powerup_type == int(PowerupType.REVERSE) and self.balls",
            "self.backward_count = self.config.reverse_powerup_ticks",
            "self.powerup_cooldown_times[powerup_type] = "
            "self.native_game_time",
            "self.powerup_triggered = True",
        ),
        "_begin_ball_explosion": (
            "if ball.exploding:",
            "return 0",
            "self.last_powerup_waypoint = int(ball.waypoint)",
            "powerup_type = self._effective_ball_powerup(ball)",
            "ball.exploding = True",
            "self._trigger_ball_powerup(ball, powerup_type)",
        ),
    }
    normalized: dict[str, str] = {}
    for name, fragments in required.items():
        normalized[name] = _verify_normalized_ast_fragments(
            functions[name],
            fragments,
            reason=f"python_powerup_{name}_contract_mismatch",
        )
    priority_fragments = required["_effective_ball_powerup"]
    priority_source = normalized["_effective_ball_powerup"]
    priority_positions = [
        priority_source.index(fragment) for fragment in priority_fragments
    ]
    if priority_positions != sorted(priority_positions):
        _fail("python_powerup_effective_type_priority_mismatch")

    begin = functions["_begin_ball_explosion"]
    guard_lines = [
        node.lineno
        for node in ast.walk(begin)
        if isinstance(node, ast.If)
        and _is_attribute(node.test, "ball", "exploding")
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Return)
        and isinstance(node.body[0].value, ast.Constant)
        and node.body[0].value.value == 0
    ]
    waypoint_lines = [
        node.lineno
        for node in ast.walk(begin)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _attribute_chain(node.targets[0])
        == ("self", "last_powerup_waypoint")
    ]
    effective_lines = [
        node.lineno
        for node in ast.walk(begin)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _is_name(node.targets[0], "powerup_type")
        and _is_call_chain(
            node.value,
            ("self", "_effective_ball_powerup"),
        )
    ]
    exploding_lines = [
        node.lineno
        for node in ast.walk(begin)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _attribute_chain(node.targets[0]) == ("ball", "exploding")
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    ]
    trigger_lines = [
        node.lineno
        for node in ast.walk(begin)
        if _is_call_chain(node, ("self", "_trigger_ball_powerup"))
    ]
    ordered_groups = (
        guard_lines,
        waypoint_lines,
        effective_lines,
        exploding_lines,
        trigger_lines,
    )
    if any(len(group) != 1 for group in ordered_groups):
        _fail("python_powerup_explosion_order_node_mismatch")
    ordered_lines = [group[0] for group in ordered_groups]
    if ordered_lines != sorted(ordered_lines):
        _fail("python_powerup_explosion_order_mismatch")

    config_matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.name == "RevengePhysicsConfig"
    ]
    if len(config_matches) != 1:
        _fail("python_powerup_config_class_mismatch")
    config = config_matches[0]
    expected_defaults = {
        "proximity_bomb_collision_pad": 56,
        "reverse_powerup_ticks": 300,
        "slow_powerup_ticks": 800,
        "slow_powerup_replace_threshold": 1_000,
    }
    found_defaults: dict[str, Any] = {}
    for node in config.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in expected_defaults
            and isinstance(node.value, ast.Constant)
        ):
            found_defaults[node.target.id] = node.value.value
    if found_defaults != expected_defaults:
        _fail("python_powerup_config_defaults_mismatch")
    config_segment = ast.get_source_segment(text, config)
    if config_segment is None:
        _fail("python_powerup_config_source_segment_missing")

    return {
        "implementation": "simulator_powerup_effect_core",
        "path": str(path),
        "functions": {
            name: _python_function_receipt(text, function)
            for name, function in functions.items()
        },
        "config": {
            "class": config.name,
            "start_line": config.lineno,
            "end_line": config.end_lineno,
            "normalized_ast_sha256": _sha256_bytes(
                ast.dump(config, include_attributes=False).encode("utf-8")
            ),
            "source_segment_sha256": _sha256_bytes(
                config_segment.encode("utf-8")
            ),
            "verified_defaults": expected_defaults,
        },
        "effective_type_priority": [
            "primary",
            "live_previous",
            "secondary",
        ],
        "last_waypoint_write": (
            "every_new_explosion_before_powerup_dispatch"
        ),
        "proximity_bomb_scope": "single_curve_active_ball_list",
        "verdict": "matches_retail_revenge_powerup_effect_core",
    }


def verify_python_terminal_contract(path: Path) -> Mapping[str, Any]:
    """Verify normal non-boss win/loss predicates in the simulator."""

    path = path.resolve(strict=True)
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise SourceGuidanceError("python_terminal_source_unreadable") from error

    functions = {
        name: _find_function(tree, name)
        for name in (
            "_front_segment_is_sucking",
            "_terminal_empty",
            "_check_outcome",
            "tick",
        )
    }
    required = {
        "_front_segment_is_sucking": (
            "for index in range(len(self.balls) - 1, -1, -1):",
            "if ball.suck_count > 0:",
            "if index == 0 or not self.balls[index - 1].contact_next:",
        ),
        "_terminal_empty": (
            "self.gun_state is not GunState.FIRING",
            "not self.free_projectiles",
            "not state.merging_projectiles",
            "not state.balls",
            "not state.pending_colors",
        ),
        "_check_outcome": (
            "if terminal_empty_at_tick_start:",
            "self.win_pending = True",
            "self.gun_state is GunState.FIRING or self.free_projectiles",
            "bool(getattr(self.curve, 'die_at_end', True))",
            "self.balls[-1].waypoint >= self.curve.end_waypoint",
            "not self._have_sets",
            "self.backward_count == 0",
            "not self._front_segment_is_sucking()",
            "self._arm_skull_entry()",
        ),
        "tick": (
            "terminal_empty_at_tick_start = self._terminal_empty()",
            "self._tick_all_curves()",
            "self._check_outcome(terminal_empty_at_tick_start="
            "terminal_empty_at_tick_start)",
        ),
    }
    for name, fragments in required.items():
        _verify_normalized_ast_fragments(
            functions[name],
            fragments,
            reason=f"python_terminal_{name}_contract_mismatch",
        )

    tick = functions["tick"]
    ordered_lines: dict[str, int] = {}
    for node in ast.walk(tick):
        if isinstance(node, ast.Assign) and any(
            _is_name(target, "terminal_empty_at_tick_start")
            for target in node.targets
        ):
            ordered_lines["sample_empty"] = node.lineno
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if _is_name(node.func.value, "self") and node.func.attr in {
                "_tick_all_curves",
                "_check_outcome",
            }:
                ordered_lines[node.func.attr] = node.lineno
    if tuple(ordered_lines) == () or not (
        ordered_lines.get("sample_empty", -1)
        < ordered_lines.get("_tick_all_curves", -1)
        < ordered_lines.get("_check_outcome", -1)
    ):
        _fail("python_terminal_tick_order_mismatch")

    return {
        "implementation": "simulator_terminal_predicates",
        "path": str(path),
        "functions": {
            name: _python_function_receipt(text, function)
            for name, function in functions.items()
        },
        "curve_win_scope": "normal_nonboss_levels",
        "special_or_boss_actor_gate_modeled": False,
        "curve_loss_predicate": "matches_retail_curve_predicate",
        "board_terminal_timing": "natural_trajectory_bound",
        "verdict": (
            "matches_normal_nonboss_predicates_with_trajectory_bound_timing"
        ),
    }


def derive_source_guidance_report(
    *,
    circleshoot_root: Path,
    runtime_executable: Path,
    simulator_source: Path,
    mechanism_audit_source: Path,
) -> Mapping[str, Any]:
    """Build the pinned Deluxe-source to Revenge-runtime comparison report."""

    circleshoot = verify_circleshoot_source(circleshoot_root)
    runtime_executable = runtime_executable.resolve(strict=True)
    try:
        runtime = runtime_executable.read_bytes()
    except OSError as error:
        raise SourceGuidanceError("runtime_unreadable") from error
    runtime_digest = _sha256_bytes(runtime)
    if len(runtime) != EXPECTED_RETAIL_RUNTIME_BYTES:
        _fail("runtime_size_mismatch")
    if runtime_digest != EXPECTED_RETAIL_RUNTIME_SHA256:
        _fail("runtime_sha256_mismatch")
    retail_gap_shot = verify_retail_gap_shot_static(runtime)
    retail_pending_color = verify_retail_pending_color_static(runtime)
    retail_rollback = verify_retail_rollback_static(runtime)
    retail_powerup_effects = verify_retail_powerup_effects_static(runtime)
    retail_terminal = verify_retail_terminal_static(runtime)
    implementations = [
        verify_python_gap_contract(
            simulator_source,
            function_name="_check_gap_shot",
            implementation="simulator",
        ),
        verify_python_gap_contract(
            mechanism_audit_source,
            function_name="_simulate_gap_checks",
            implementation="mechanism_audit",
        ),
        verify_python_pending_color_contract(simulator_source),
        verify_python_rollback_contract(simulator_source),
        verify_python_powerup_effect_contract(simulator_source),
        verify_python_terminal_contract(simulator_source),
    ]
    return {
        "schema": SOURCE_GUIDANCE_SCHEMA,
        "version": SOURCE_GUIDANCE_VERSION,
        "status": "PASS",
        "classification": (
            "ancestor_source_deltas_resolved_by_pinned_target_runtime"
        ),
        "ancestor_source": circleshoot,
        "target_runtime": {
            "path": str(runtime_executable),
            "bytes": len(runtime),
            "sha256": runtime_digest,
            "gap_shot": retail_gap_shot,
            "pending_color": retail_pending_color,
            "rollback": retail_rollback,
            "powerup_effects": retail_powerup_effects,
            "terminal": retail_terminal,
        },
        "comparisons": {
            "gap_shot_curve_proximity": {
                "shared_sample_step": "2 * projectile_radius",
                "ancestor_threshold": "(2 * projectile_radius) ** 2",
                "target_threshold": "projectile_radius ** 2",
                "target_authority": "pinned_retail_zumas_revenge_runtime",
                "decision": "retain_radius_squared_threshold",
                "copy_ancestor_implementation_verbatim": False,
            },
            "pending_color_rng": {
                "shared_candidate_rejection": (
                    "repeat while candidate == previous_color"
                ),
                "ancestor_repeat_guard": "repeat_roll <= repeat_chance",
                "target_repeat_guard": (
                    "repeat_roll <= repeat_chance and "
                    "current_run < max_clump"
                ),
                "ancestor_visual_frame_order": "before_color_selection",
                "target_visual_frame_order": (
                    "after_color_selection_before_list_insert"
                ),
                "target_authority": "pinned_retail_zumas_revenge_runtime",
                "decision": "retain_target_rng_order_and_max_clump_guard",
                "copy_ancestor_implementation_verbatim": False,
            },
            "rollback_chain": {
                "shared_topology": [
                    "connected_backward_motion",
                    "30_tick_gap_contact_rollback",
                    "10_count_set_removal_suck_seed",
                    "40_tick_entrance_stop",
                ],
                "ancestor_suck_ramp": "integer_divide_by_8",
                "target_suck_ramp": (
                    "arithmetic_shift_right_3_then_multiply_reverse_speed"
                ),
                "ancestor_direction_flag": False,
                "target_direction_flag": True,
                "ancestor_global_reverse_speed": "hardcoded_1.0",
                "target_global_reverse_speed": "curve_reverse_speed",
                "target_authority": "pinned_retail_zumas_revenge_runtime",
                "decision": (
                    "scale_suckback_and_explicitly_restore_backward_direction"
                ),
                "copy_ancestor_implementation_verbatim": False,
            },
            "powerup_effects": {
                "shared_effects": {
                    "proximity_bomb": (
                        "active_chain_physical_collision_then_recursive_clear"
                    ),
                    "reverse_ticks": 300,
                    "slow_threshold": 1_000,
                    "slow_ticks": 800,
                    "per_ball_waypoint_update": True,
                },
                "ancestor_bomb_collision_pad": 45,
                "target_bomb_collision_pad": 56,
                "target_effective_type_priority": [
                    "primary",
                    "live_previous",
                    "secondary",
                ],
                "target_board_scope": "all_curves",
                "target_authority": "pinned_retail_zumas_revenge_runtime",
                "decision": (
                    "use_target_pad_and_update_last_waypoint_on_every_new_"
                    "explosion"
                ),
                "spawn_scheduler_resolved": False,
                "fruit_collision_resolved": False,
                "copy_ancestor_implementation_verbatim": False,
            },
            "natural_terminal_predicates": {
                "shared_win_predicate": [
                    "active_ball_list_empty",
                    "pending_ball_list_empty",
                ],
                "target_additional_win_guard": (
                    "special_or_boss_actor_absent"
                ),
                "shared_loss_predicate_core": [
                    "no_active_sets",
                    "front_ball_at_endpoint",
                    "no_curve_bullets",
                    "no_global_reverse",
                    "no_front_segment_suction",
                ],
                "target_additional_loss_guard": "curve_dies_at_endpoint",
                "target_authority": "pinned_retail_zumas_revenge_runtime",
                "decision": (
                    "retain_normal_nonboss_predicates_and_leave_board_timing_"
                    "trajectory_bound"
                ),
                "boss_actor_gate_modeled": False,
                "copy_ancestor_implementation_verbatim": False,
            },
        },
        "python_implementations": implementations,
        "scope": {
            "fidelity_gate_credit": False,
            "training_authorized": False,
            "reason": (
                "static source guidance prevents implementation drift but "
                "does not replace natural retail trajectory evidence"
            ),
        },
    }


def write_source_guidance_report(
    report: Mapping[str, Any],
    output: Path,
) -> None:
    """Write one immutable canonical JSON report."""

    data = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    output = output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        _fail("output_must_be_absent_with_existing_parent")
    try:
        with output.open("xb") as stream:
            stream.write(data)
    except OSError as error:
        raise SourceGuidanceError("output_write_failed") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circleshoot-root", required=True, type=Path)
    parser.add_argument("--runtime-executable", required=True, type=Path)
    parser.add_argument("--simulator-source", required=True, type=Path)
    parser.add_argument("--mechanism-audit-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = derive_source_guidance_report(
        circleshoot_root=args.circleshoot_root,
        runtime_executable=args.runtime_executable,
        simulator_source=args.simulator_source,
        mechanism_audit_source=args.mechanism_audit_source,
    )
    write_source_guidance_report(report, args.output)
    print(args.output.resolve())
    print(_sha256_path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
