from pathlib import Path
import pickle
import pandas as pd

from src.config import RELPATH, FILENAME
import src.instances as instances
import src.main_extb as main_extb
import src.solver_compare as solver_compare


# =========================
# User-editable experiment plan
# =========================
N_PLAYERS = 1430
OBJECTIVES = ["maximin", "utilitarian"]

TIME_LIMIT = 180          # seconds for blocking MILP
EPS_LIMIT = 1.0           # multiplicative least objection threshold
ITER_LIMIT = 10
K_CAPS = [None, 2, 5, 10, 20, 50, 100]

# Optional coalition-cost threshold in multiplicative terms.
# Keep this at 1.0 for now unless you want the "coalition formation cost" variant.
MIN_BLOCK_GAIN_MULT = 1.0


def ensure_dirs() -> None:
    Path(RELPATH, "results", "instances").mkdir(parents=True, exist_ok=True)
    Path(RELPATH, "results", "solutions").mkdir(parents=True, exist_ok=True)
    Path(RELPATH, "results", "figures").mkdir(parents=True, exist_ok=True)
    Path(RELPATH, "results", "summary").mkdir(parents=True, exist_ok=True)


def ensure_instance(n: int) -> None:
    instance_path = Path(RELPATH, "results", "instances", f"instance_{FILENAME}_{n}.pkl")
    if not instance_path.exists():
        print(f"Instance file not found, generating it for n={n} ...")
        instances.main(n=n)
    else:
        print(f"Using existing instance: {instance_path}")


def summarize_solution(n, objective, coalition_size_cap, min_block_gain_mult):
    modelname = solver_compare.build_modelname(
        n, objective, TIME_LIMIT, EPS_LIMIT,
        coalition_size_cap=coalition_size_cap,
        min_block_gain_mult=min_block_gain_mult
    )

    latest_iter = None
    solutions_dir = Path(RELPATH, "results", "solutions")

    for pkl_path in solutions_dir.glob(f"{FILENAME}_{modelname}_*.pkl"):
        iter_str = pkl_path.stem.rsplit("_", 1)[-1]
        try:
            iter_val = int(iter_str)
        except ValueError:
            continue
        latest_iter = iter_val if latest_iter is None else max(latest_iter, iter_val)

    if latest_iter is None:
        return None

    sol_path = solutions_dir / f"{FILENAME}_{modelname}_{latest_iter}.pkl"
    with open(sol_path, "rb") as f:
        x_N, u_N, tt, cutct, eps, S, kappa = pickle.load(f)

    util_welfare = sum(u_N.values()) / len(u_N)
    maximin_welfare = min(u_N.values())
    sorted_utils = sorted(u_N.values())
    median_utility = sorted_utils[len(sorted_utils) // 2]

    return {
        "modelname": modelname,
        "n": n,
        "objective": objective,
        "k_cap": coalition_size_cap if coalition_size_cap is not None else "full",
        "min_block_gain_mult": min_block_gain_mult,
        "final_iter": latest_iter,
        "elapsed_seconds": tt,
        "cut_count": cutct,
        "eps_final": eps,
        "kappa_final": kappa,
        "utilitarian_welfare": util_welfare,
        "maximin_welfare": maximin_welfare,
        "median_utility": median_utility,
        "min_utility": min(u_N.values()),
        "max_utility": max(u_N.values()),
        "blocking_size_final": len(S) if S is not None else None,
        "active_lines": sum(1 for val in x_N.values() if val > 1e-9),
    }


def main():
    ensure_dirs()
    ensure_instance(N_PLAYERS)

    rows = []

    for objective in OBJECTIVES:
        for k_cap in K_CAPS:
            print("=" * 90)
            print(f"Running objective={objective}, k_cap={k_cap}, gain_mult={MIN_BLOCK_GAIN_MULT}")

            _, meta = main_extb.main(
                N_PLAYERS,
                objective,
                TIME_LIMIT,
                EPS_LIMIT,
                iterLimit=ITER_LIMIT,
                coalition_size_cap=k_cap,
                min_block_gain_mult=MIN_BLOCK_GAIN_MULT,
            )

            print(
                f"Finished {meta['modelname']} "
                f"at iter {meta['iterCount']} with eps={meta['eps']}"
            )

            row = summarize_solution(
                N_PLAYERS,
                objective,
                coalition_size_cap=k_cap,
                min_block_gain_mult=MIN_BLOCK_GAIN_MULT
            )
            if row is not None:
                rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_csv = Path(RELPATH, "results", "summary", f"extB_summary_{FILENAME}_n{N_PLAYERS}.csv")
    summary_df.to_csv(out_csv, index=False)

    print(f"\nSaved summary to: {out_csv}")
    print(summary_df)


if __name__ == "__main__":
    main()