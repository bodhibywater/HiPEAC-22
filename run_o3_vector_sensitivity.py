"""gem5 v25.1 RISC-V O3 runner with a configurable generic SIMD timing model.

This is an architectural sensitivity experiment, NOT a calibrated physical RVV core.
Run from gem5's build/RISCV/gem5.opt using the same workloads as run_o3.py.
"""

import argparse
from pathlib import Path

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires
from m5.objects.FUPool import DefaultFUPool
from m5.objects.FuncUnitConfig import SIMD_Unit


def configure_simd_execution(core, simd_units: int, simd_latency: int) -> None:
    """Replace the default O3 FU pool with one tuned for SIMD sensitivity tests.

    Every operation in the *generic* SIMD_Unit gets the same result latency.
    This does NOT configure RVV load/store ports, lane count or datapath width.
    """
    pool = DefaultFUPool()
    simd_descriptors = [fu for fu in pool.FUList if isinstance(fu, SIMD_Unit)]
    if len(simd_descriptors) != 1:
        raise RuntimeError("Expected exactly one SIMD_Unit in DefaultFUPool")

    simd = simd_descriptors[0]
    simd.count = simd_units
    for operation in simd.opList:
        operation.opLat = simd_latency

    core.instQueues[0].fuPool = pool
    print(f"SIMD_MODEL units={simd_units} operation_latency={simd_latency} cycles")


def main() -> None:
    """Run the same RISC-V O3 workload with configurable SIMD FU assumptions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--vlen", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--simd-units", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--simd-latency", type=int, choices=(1, 3, 4, 5), default=1)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    binary = Path(args.binary).resolve(strict=True)
    requires(isa_required=ISA.RISCV)
    processor = SimpleProcessor(cpu_type=CPUTypes.O3, isa=ISA.RISCV, num_cores=1)
    for wrapped_core in processor.cores:
        core = wrapped_core.core
        core.isa[0].enable_rvv = True
        core.isa[0].vlen = args.vlen
        configure_simd_execution(core, args.simd_units, args.simd_latency)

    board = SimpleBoard(
        clk_freq="2GHz",
        processor=processor,
        memory=SingleChannelDDR3_1600(size="1GiB"),
        cache_hierarchy=PrivateL1PrivateL2CacheHierarchy(
            l1d_size="32KiB", l1i_size="32KiB", l2_size="512KiB"
        ),
    )
    binary_args = args.args[1:] if args.args and args.args[0] == "--" else args.args
    board.set_se_binary_workload(
        binary=BinaryResource(local_path=str(binary), architecture=ISA.RISCV),
        arguments=binary_args,
    )
    Simulator(board=board).run()


if __name__ in ("__main__", "__m5_main__"):
    main()
