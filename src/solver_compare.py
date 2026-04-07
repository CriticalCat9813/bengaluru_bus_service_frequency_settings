import pickle
import time
import warnings

import gurobipy as gp
import numpy as np
import scipy.sparse as ss

from src.config import *
from src.solver import get_basis, get_intersections


def main(instance, modelname, **kwargs):
    """
    Unified solver for:
      - baseline paper method: coalition_size_cap=None
      - Extension B (k-core): coalition_size_cap = k
      - optional coalition-cost threshold: min_block_gain_mult > 1.0

    The implementation keeps the paper/code's multiplicative least objection
    formulation and individual-rationality cuts, then modifies the blocking
    problem by constraining sum_i y_i <= k when coalition_size_cap is given.
    """

    N, J, K, A, B, V = instance

    ts = time.time()
    slackCount = 0
    cutCount = 0
    iterCount = -1
    eps, S = -1, None

    coalition_size_cap = kwargs.get("coalition_size_cap", None)
    min_block_gain_mult = kwargs.get("min_block_gain_mult", 1.0)

    m = gp.Model()
    m.Params.OutputFlag = kwargs.get('OutputFlag', 1)
    m.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-6)
    m.Params.Method = kwargs.get('Method', 1)
    m.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m.Params.OptimalityTol = kwargs.get('OptimalityTol', 1E-9)
    m.ModelSense = -1

    m._x = m.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m._u = m.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='u')

    for k in K:
        s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name=f's[{slackCount}]')
        slackCount += 1
        m.addConstr(gp.quicksum(A[k][j] * m._x[j] for j in J) + s == sum(B[i][k] for i in N))

    for i in N:
        s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name=f's[{slackCount}]')
        slackCount += 1
        m.addConstr(m._u[i] + s == gp.quicksum(V[i][j] * m._x[j] for j in J))

    objective = kwargs.get('objective', 'utilitarian')
    if objective == 'utilitarian':
        m.setObjective((gp.quicksum(m._u[i] for i in N)) / len(N))
    elif objective == 'maximin':
        z = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='z')
        for i in N:
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name=f's[{slackCount}]')
            slackCount += 1
            m.addConstr(z + s == m._u[i])
        m.setObjective(z + 1E-6 * (gp.quicksum(m._u[i] for i in N)) / len(N))
    else:
        raise ValueError(f'objective {objective} not supported')

    # Initial solve without any cooperation cuts.
    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}
    kappa = m.getAttr('KappaExact')

    out = x_N, u_N, time.time() - ts, cutCount, eps, S, kappa
    with open(f'{RELPATH}/results/solutions/{FILENAME}_{modelname}_{iterCount}.pkl', 'wb') as file:
        pickle.dump(out, file)

    # Paper implementation adds singleton/individual-rationality constraints.
    if kwargs.get('indRat', True):
        for i in N:
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name=f's[{slackCount}]')
            slackCount += 1
            m.addConstr(m._u[i] - s == max(V[i][j] / A[0][j] for j in J))
            cutCount += 1

    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}
    kappa = m.getAttr('KappaExact')

    timeLimit = kwargs.get('timeLimit', 60)
    iterLimit = kwargs.get('iterLimit', 100)
    epsLimit = kwargs.get('epsLimit', 1.0)
    default_starts = {tuple(sorted(N))} if coalition_size_cap is None else set()
    Starts = kwargs.get('Starts', default_starts).copy()

    iterCount += 1
    eps, S = get_blocking(
        instance, u_N,
        TimeLimit=timeLimit,
        Starts=Starts,
        coalition_size_cap=coalition_size_cap,
        min_block_gain_mult=min_block_gain_mult,
    )
    S = tuple(sorted(S))

    out = x_N, u_N, time.time() - ts, cutCount, eps, S, kappa
    with open(f'{RELPATH}/results/solutions/{FILENAME}_{modelname}_{iterCount}.pkl', 'wb') as file:
        pickle.dump(out, file)

    while eps > epsLimit and iterCount <= iterLimit:
        print(f'iterCount: {iterCount}')

        print('... extracting basis.')
        constr_names_to_indices = {constr.ConstrName: i for i, constr in enumerate(m.getConstrs())}
        basis_mat, basis_varnames = get_basis(m, constr_names_to_indices)
        print(f'...... extracted basis of shape {basis_mat.shape}.')
        with warnings.catch_warnings():
            warnings.simplefilter("error", ss.linalg.MatrixRankWarning)
            try:
                _ = ss.linalg.spsolve(basis_mat, np.zeros(basis_mat.shape[0]))
            except ss.linalg.MatrixRankWarning:
                ss.save_npz(
                    f'{RELPATH}/results/solutions/{FILENAME}_{modelname}_{iterCount}.npz',
                    basis_mat
                )
                print('......... stored seemingly singular basis.')
        basis_mat_lu = ss.linalg.splu(basis_mat.tocsc())

        print('... adding cuts for previous S.')
        cutPrev = False
        setS = set(S)
        for prev_S in Starts:
            if not setS.isdisjoint(set(prev_S)):
                continue
            intersections = get_intersections(
                instance, m, constr_names_to_indices, basis_mat_lu, basis_varnames, u_N, prev_S,
                epsTh=0, lamRatTh=1E-6
            )
            if intersections is not None:
                cutPrev = True
                min_lam = min(lam for _, lam in intersections)
                max_lam = max(lam for _, lam in intersections)
                s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name=f's[{slackCount}]')
                slackCount += 1
                m.addConstr(gp.quicksum(m.getVarByName(varname) / lam for varname, lam in intersections) - s == 1)
                cutCount += 1
                print(f'...... added cut for prev. S with coeff. ratio {min_lam / max_lam}.')

        print(f'... adding cut for current S of length {len(S)}.')
        intersections = get_intersections(
            instance, m, constr_names_to_indices, basis_mat_lu, basis_varnames, u_N, S,
            epsTh=0, lamRatTh=0
        )
        if intersections is not None:
            min_lam = min(lam for _, lam in intersections)
            max_lam = max(lam for _, lam in intersections)
            if min_lam / max_lam >= 1E-6 or not cutPrev:
                s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name=f's[{slackCount}]')
                slackCount += 1
                m.addConstr(gp.quicksum(m.getVarByName(varname) / lam for varname, lam in intersections) - s == 1)
                cutCount += 1
                print(f'...... added cut for curr. S with coeff. ratio {min_lam / max_lam}.')

        Starts.add(S)
        print('... solving model.')
        m.optimize()
        x_N = {j: m._x[j].X for j in J}
        u_N = {i: m._u[i].X for i in N}
        kappa = m.getAttr('KappaExact')

        iterCount += 1
        eps, S = get_blocking(
            instance, u_N,
            TimeLimit=timeLimit,
            Starts=Starts,
            divPhase=False,
            coalition_size_cap=coalition_size_cap,
            min_block_gain_mult=min_block_gain_mult,
        )
        S = tuple(sorted(S))

        out = x_N, u_N, time.time() - ts, cutCount, eps, S, kappa
        with open(f'{RELPATH}/results/solutions/{FILENAME}_{modelname}_{iterCount}.pkl', 'wb') as file:
            pickle.dump(out, file)

    meta = {
        "iterCount": iterCount,
        "modelname": modelname,
        "coalition_size_cap": coalition_size_cap,
        "min_block_gain_mult": min_block_gain_mult,
        "objective": objective,
        "eps": eps,
    }
    return out, meta


def get_blocking(instance, u_N, **kwargs):
    """
    Blocking coalition search under multiplicative least objection.

    Extra kwargs:
      coalition_size_cap: if given, impose sum_i y_i <= k.
      min_block_gain_mult: require eps >= this threshold to count as a credible block.
                           baseline paper implementation corresponds to 1.0.
    """
    N, J, K, A, B, V = instance

    coalition_size_cap = kwargs.get("coalition_size_cap", None)
    min_block_gain_mult = kwargs.get("min_block_gain_mult", 1.0)

    default_starts = {tuple(sorted(N))} if coalition_size_cap is None else set()
    Starts = kwargs.get('Starts', default_starts).copy()
    if coalition_size_cap is not None:
        Starts = {S for S in Starts if len(S) <= coalition_size_cap}

    startCts = {i: 0 for i in N}
    for Start in Starts:
        for i in Start:
            startCts[i] += 1
    weights = {i: np.exp2(-startCts[i]) for i in N}
    max_weight = max(weights.values()) if weights else 1.0
    for i in N:
        weights[i] /= max_weight

    # Heuristic starts from single-line blocks, filtered by k if applicable.
    for j in J:
        eps_j, S_j = 1.0, None
        A_j = A[0][j]
        N_j, V_j = zip(*sorted([(i, V[i][j]) for i in N], key=lambda val: val[1], reverse=True))
        max_ct = len(N_j) if coalition_size_cap is None else min(len(N_j), coalition_size_cap)
        for Ct in range(1, max_ct + 1):
            if V_j[Ct - 1] <= 0:
                break
            eps = min(V_ij * Ct / A_j / u_N[i] for i, V_ij in zip(N_j[:Ct], V_j[:Ct]))
            if eps > eps_j:
                eps_j = eps
                S_j = tuple(sorted(i for i in N_j[:Ct]))
        if S_j is not None:
            Starts.add(S_j)

    m_S = gp.Model()
    m_S.Params.BestObjStop = kwargs.get('BestObjStop', m_S.Params.BestObjStop)
    m_S.Params.OutputFlag = kwargs.get('OutputFlag', 1)
    m_S.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-6)
    m_S.Params.MIPFocus = kwargs.get('MIPFocus', 1)
    m_S.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m_S.NumStart = len(Starts)
    m_S.ModelSense = -1

    m_S._del = m_S.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='del')
    m_S._eps = m_S.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='eps')
    m_S._y = m_S.addVars(N, vtype=gp.GRB.BINARY, name='y')
    m_S._x = m_S.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m_S._u = m_S.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='u')

    m_S.addConstr(gp.quicksum(m_S._y[i] for i in N) >= 1)
    if coalition_size_cap is not None:
        m_S.addConstr(gp.quicksum(m_S._y[i] for i in N) <= coalition_size_cap, name='k_cap')

    for k in K:
        m_S.addConstr(gp.quicksum(A[k][j] * m_S._x[j] for j in J) == gp.quicksum(B[i][k] * m_S._y[i] for i in N))
    for i in N:
        m_S.addConstr(m_S._u[i] == gp.quicksum(V[i][j] * m_S._x[j] for j in J))
    for i in N:
        m_S.addGenConstrIndicator(m_S._y[i], True, m_S._eps * u_N[i] - m_S._u[i], gp.GRB.LESS_EQUAL, 0)
        m_S.addGenConstrIndicator(m_S._y[i], True, m_S._del - m_S._u[i], gp.GRB.LESS_EQUAL, -u_N[i])

    # Paper implementation's additive floor for numerical stability.
    m_S.addConstr(m_S._del >= 1E-3)

    # Optional coalition-formation-cost threshold in multiplicative terms.
    if min_block_gain_mult > 1.0:
        m_S.addConstr(m_S._eps >= min_block_gain_mult, name='min_gain_mult')

    valid_starts = [S for S in Starts if coalition_size_cap is None or len(S) <= coalition_size_cap]
    for StartNumber, Start in enumerate(valid_starts):
        m_S.Params.StartNumber = StartNumber
        for i in N:
            m_S._y[i].Start = 1 if i in Start else 0

    if kwargs.get('divPhase', False):
        m_S.Params.TimeLimit = kwargs.get('TimeLimit', 300)
        m_S.setObjective(m_S._del)
        m_S.optimize()

        if m_S.SolCount > 0:
            m_S.NumStart += 1
            Start = {i for i in N if m_S._y[i].X > 1 / 2}
            if coalition_size_cap is None or len(Start) <= coalition_size_cap:
                m_S.Params.StartNumber += 1
                for i in N:
                    m_S._y[i].Start = 1 if i in Start else 0
            m_S.addConstr(m_S._del >= 1 + (m_S._del.X - 1) / 2)

    m_S.Params.TimeLimit = kwargs.get('TimeLimit', 300)
    m_S.setObjective(m_S._eps)
    m_S.optimize()

    if m_S.SolCount == 0:
        return 1.0, set()

    eps = m_S._eps.X
    S = {i for i in N if m_S._y[i].X > 1 / 2}
    return eps, S


def build_modelname(n, objective, timeLimit, epsLimit, coalition_size_cap=None, min_block_gain_mult=1.0):
    parts = [str(n), objective, str(timeLimit), str(epsLimit)]
    if coalition_size_cap is None:
        parts.append("fullcore")
    else:
        parts.append(f"k{coalition_size_cap}")
    if min_block_gain_mult > 1.0:
        gain_tag = str(min_block_gain_mult).replace(".", "p")
        parts.append(f"gain{gain_tag}")
    return "-".join(parts)
