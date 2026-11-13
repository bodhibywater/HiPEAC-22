"""Experimental gem5 RISC-V SE-mode runner (NOT validated in this environment).

Designed against the gem5 stdlib API around v25.1. Verify simulator version,
RVV configuration and statistics before relying on any architectural results.

Example:
  build/RISCV/gem5.opt --outdir=m5out/sparse gem5/run_se.py \\
      --binary /abs/path/snn_bench -- --mode sparse --repeats 1 --warmup 0
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


def main() -> None:
    """Run one statically linked RISC-V benchmark in a fixed timing system."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--vlen", type=int, default=256)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.vlen not in (128, 256, 512):
        parser.error("Supported experiment VLEN values: 128, 256, 512")
    path = Path(args.binary).resolve(strict=True)
    requires(isa_required=ISA.RISCV)
    processor = SimpleProcessor(cpu_type=CPUTypes.TIMING, isa=ISA.RISCV, num_cores=1)
    for core in processor.cores:
        core.core.isa[0].enable_rvv = True
        core.core.isa[0].vlen = args.vlen
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
        binary=BinaryResource(local_path=str(path), architecture=ISA.RISCV),
        arguments=binary_args,
    )
    Simulator(board=board).run()


if __name__ == "__main__":
    main()
