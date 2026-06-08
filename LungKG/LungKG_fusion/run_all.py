#!/usr/bin/env python3
"""
LungKG Fusion Pipeline — Run all 6 steps with one command
Usage:
    python run_all.py                  # Run all steps 1-6
    python run_all.py --step 3         # Run only step 3
    python run_all.py --step 2 --to 5  # Run steps 2 to 5
"""

import argparse
import time
import sys
import traceback


def run_step(step_num: int):
    """Run the specified step and return whether it succeeded"""
    step_map = {
        1: ("Data Parsing & Standardization", "step1_parse_and_standardize"),
        2: ("Entity Alignment",                "step2_entity_alignment"),
        3: ("Deduplication",                   "step3_deduplication"),
        4: ("Relation Completion",             "step4_relation_completion"),
        5: ("Quality Check",                   "step5_quality_check"),
        6: ("Neo4j Export",                    "step6_export_neo4j"),
    }

    if step_num not in step_map:
        print(f"[ERROR] Invalid step number: {step_num}, valid range 1-6")
        return False

    name, module_name = step_map[step_num]
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {name}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        mod = __import__(module_name)
        import asyncio
        if hasattr(mod, "main"):
            result = mod.main()
            # If main returns a coroutine, execute it with asyncio.run
            if asyncio.iscoroutine(result):
                asyncio.run(result)
        elif hasattr(mod, "run"):
            result = mod.run()
            if asyncio.iscoroutine(result):
                asyncio.run(result)
        else:
            print(f"[WARN] Module {module_name} has no main() or run() function, import only")
    except Exception:
        traceback.print_exc()
        elapsed = time.time() - t0
        print(f"\n[FAILED] Step {step_num} failed, elapsed {elapsed:.1f}s")
        return False

    elapsed = time.time() - t0
    print(f"\n[OK] Step {step_num} completed, elapsed {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="LungKG Fusion Pipeline One-Click Run")
    parser.add_argument("--step", type=int, default=None,
                        help="Starting step number (1-6), defaults to 1")
    parser.add_argument("--to", type=int, default=None,
                        help="Ending step number (1-6), defaults to 6")
    args = parser.parse_args()

    start = args.step if args.step is not None else 1
    end = args.to if args.to is not None else (args.step if args.step is not None and args.to is None else 6)
    # If only --step is specified without --to, run only that step
    if args.step is not None and args.to is None:
        end = start

    if not (1 <= start <= 6 and 1 <= end <= 6 and start <= end):
        print(f"[ERROR] Invalid step range: {start}-{end}, valid range 1-6")
        sys.exit(1)

    print(f"LungKG Fusion Pipeline: run steps {start} -> {end}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_t0 = time.time()
    failed = []

    for step_num in range(start, end + 1):
        ok = run_step(step_num)
        if not ok:
            failed.append(step_num)
            print(f"\n[ABORT] Step {step_num} failed, aborting subsequent steps.")
            break

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  All done! Total elapsed {total_elapsed:.1f}s")
    if failed:
        print(f"  Failed steps: {failed}")
        sys.exit(1)
    else:
        print(f"  Steps {start}-{end} all succeeded ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
