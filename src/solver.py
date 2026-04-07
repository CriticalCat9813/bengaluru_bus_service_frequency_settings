import gurobipy as gp
import numpy as np
import pickle
import scipy.sparse as ss
import time
import warnings
from src.config import *


def main(instance, modelname, **kwargs):

    N, J, K, A, B, V = instance

    ts = time.time()
    slackCount = 0
    cutCount = 0
    iterCount = -1
    eps, S, = -1, None

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
        s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='s[{0}]'.format(slackCount))
        slackCount += 1
        m.addConstr(gp.quicksum(A[k][j] * m._x[j] for j in J) + s == sum(B[i][k] for i in N))
    for i in N:
        s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='s[{0}]'.format(slackCount))
        slackCount += 1
        m.addConstr(m._u[i] + s == gp.quicksum(V[i][j] * m._x[j] for j in J))

    objective = kwargs.get('objective', 'utilitarian')
    if objective == 'utilitarian':
        m.setObjective((gp.quicksum(m._u[i] for i in N))/len(N))
    elif objective == 'maximin':
        z = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='z')
        for i in N:
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='s[{0}]'.format(slackCount))
            slackCount += 1
            m.addConstr(z + s == m._u[i])
        m.setObjective(z + 1E-6*(gp.quicksum(m._u[i] for i in N))/len(N))
    else:
        raise Exception('objective {0} not supported'.format(objective))

    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}
    kappa = m.getAttr('KappaExact')

    out = x_N, u_N, time.time() - ts, cutCount, eps, S, kappa
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, iterCount), 'wb') as file:
        pickle.dump(out, file)

    if kwargs.get('indRat', True):
        for i in N:
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='s[{0}]'.format(slackCount))
            slackCount += 1
            m.addConstr(m._u[i] - s == max(V[i][j]/A[0][j] for j in J))
            cutCount += 1

    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}
    kappa = m.getAttr('KappaExact')

    timeLimit = kwargs.get('timeLimit', 60)
    iterLimit = kwargs.get('iterLimit', 10)
    epsLimit = kwargs.get('epsLimit', 0)
    Starts = {tuple(sorted(N))}

    iterCount += 1
    eps, S = get_blocking(instance, u_N, TimeLimit=timeLimit, Starts=Starts)
    S = tuple(sorted(S))

    out = x_N, u_N, time.time() - ts, cutCount, eps, S, kappa
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, iterCount), 'wb') as file:
        pickle.dump(out, file)

    while eps > epsLimit and iterCount <= iterLimit:

        print('iterCount: {0}'.format(iterCount))

        print('... extracting basis.')
        constr_names_to_indices = {constr.ConstrName: i for i, constr in enumerate(m.getConstrs())}
        basis_mat, basis_varnames = get_basis(m, constr_names_to_indices)
        print('...... extracted basis of shape {0}.'.format(basis_mat.shape))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ss.linalg.MatrixRankWarning)
            try:
                _ = ss.linalg.spsolve(basis_mat, np.zeros(basis_mat.shape[0]))
            except ss.linalg.MatrixRankWarning:
                ss.save_npz(
                    '{0}/results/solutions/{1}_{2}_{3}.npz'.format(RELPATH, FILENAME, modelname, iterCount), basis_mat
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
                s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='s[{0}]'.format(slackCount))
                slackCount += 1
                m.addConstr(gp.quicksum(m.getVarByName(varname) / lam for varname, lam in intersections) - s == 1)
                cutCount += 1
                print('...... added cut for prev. S with coeff. ratio {0}.'.format(min_lam/max_lam))
        print('... adding cut for current S of length {0}.'.format(len(S)))
        intersections = get_intersections(
            instance, m, constr_names_to_indices, basis_mat_lu, basis_varnames, u_N, S,
            epsTh=0, lamRatTh=0
        )
        if intersections is not None:
            min_lam = min(lam for _, lam in intersections)
            max_lam = max(lam for _, lam in intersections)
            if min_lam/max_lam >= 1E-6 or not cutPrev:
                s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='s[{0}]'.format(slackCount))
                slackCount += 1
                m.addConstr(gp.quicksum(m.getVarByName(varname) / lam for varname, lam in intersections) - s == 1)
                cutCount += 1
                print('...... added cut for curr. S with coeff. ratio {0}.'.format(min_lam / max_lam))
        # if iterCount % 2 == 0:
        Starts.add(S)
        print('... solving model.')

        m.optimize()
        x_N = {j: m._x[j].X for j in J}
        u_N = {i: m._u[i].X for i in N}
        kappa = m.getAttr('KappaExact')

        iterCount += 1
        # if iterCount % 2 == 0:
        #     eps, S = get_blocking(instance, u_N, TimeLimit=timeLimit, Starts=Starts, divPhase=False)
        # else:
        eps, S = get_blocking(instance, u_N, TimeLimit=timeLimit, Starts=Starts, divPhase=False)
        S = tuple(sorted(S))

        out = x_N, u_N, time.time() - ts, cutCount, eps, S, kappa
        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, iterCount), 'wb') as file:
            pickle.dump(out, file)

    return out


def get_blocking(instance, u_N, **kwargs):

    N, J, K, A, B, V = instance

    Starts = kwargs.get('Starts', {tuple(sorted(N))}).copy()

    startCts = {i: 0 for i in N}
    for Start in Starts:
        for i in Start:
            startCts[i] += 1
    weights = {i: np.exp2(-startCts[i]) for i in N}
    max_weight = max(weights.values())
    for i in N:
        weights[i] /= max_weight

    for j in J:
        eps_j, S_j = 0, None
        A_j = A[0][j]
        N_j, V_j = zip(*sorted([(i, V[i][j]) for i in N], key=lambda val: val[1], reverse=True))
        for Ct in range(1, len(N_j) + 1):
            if V_j[Ct - 1] <= 0:
                break
            eps = min(V_ij * Ct / A_j - u_N[i] for i, V_ij in zip(N_j[:Ct], V_j[:Ct]))
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
    for k in K:
        m_S.addConstr(gp.quicksum(A[k][j] * m_S._x[j] for j in J) == gp.quicksum(B[i][k] * m_S._y[i] for i in N))
    for i in N:
        m_S.addConstr(m_S._u[i] == gp.quicksum(V[i][j] * m_S._x[j] for j in J))
    for i in N:
        # m_S.addGenConstrIndicator(m_S._y[i], True, m_S._del - weights[i] * m_S._u[i], gp.GRB.LESS_EQUAL, - weights[i] * u_N[i])
        # m_S.addGenConstrIndicator(m_S._y[i], True, m_S._eps - m_S._u[i], gp.GRB.LESS_EQUAL, - u_N[i])
        # m_S.addGenConstrIndicator(m_S._y[i], True, m_S._del * weights[i] * u_N[i] - m_S._u[i], gp.GRB.LESS_EQUAL, 0)
        m_S.addGenConstrIndicator(m_S._y[i], True, m_S._eps * u_N[i] - m_S._u[i], gp.GRB.LESS_EQUAL, 0)

        m_S.addGenConstrIndicator(m_S._y[i], True, m_S._del - m_S._u[i], gp.GRB.LESS_EQUAL, - u_N[i])

    m_S.addConstr(m_S._del >= 1E-3)

    for StartNumber, Start in enumerate(Starts):
        m_S.Params.StartNumber = StartNumber
        for i in N:
            if i in Start:
                m_S._y[i].Start = 1
            else:
                m_S._y[i].Start = 0

    if kwargs.get('divPhase', False):

        m_S.Params.TimeLimit = kwargs.get('TimeLimit', 300)
        m_S.setObjective(m_S._del)
        m_S.optimize()

        m_S.NumStart += 1
        Start = {i for i in N if m_S._y[i].X > 1/2}
        m_S.Params.StartNumber += 1
        for i in N:
            if i in Start:
                m_S._y[i].Start = 1
            else:
                m_S._y[i].Start = 0

        # m_S.addConstr(m_S._del >= m_S._del.X / 2)
        m_S.addConstr(m_S._del >= 1 + (m_S._del.X - 1)/2)

    m_S.Params.TimeLimit = kwargs.get('TimeLimit', 300)
    m_S.setObjective(m_S._eps)
    m_S.optimize()

    eps = m_S._eps.X
    S = {i for i in N if m_S._y[i].X > 1/2}

    return eps, S


def get_intersections(instance, m, constr_names_to_indices, basis_mat_lu, basis_varnames, u_N, S, **kwargs):

    N, J, K, A, B, V = instance

    epsTh = kwargs.get('epsTh', 0)
    lamRatTh = kwargs.get('lamRatTh', 1E-9)

    m_S = gp.Model()
    m_S.Params.OutputFlag = kwargs.get('OutputFlag', 0)
    m_S.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-6)
    m_S.Params.NumericFocus = kwargs.get('NumericFocus', 0)
    m_S.Params.OptimalityTol = kwargs.get('OptimalityTol', 1E-6)
    m_S.ModelSense = -1

    m_S._lam = m_S.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='lam')
    m_S._x = m_S.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m_S._u = m_S.addVars(S, vtype=gp.GRB.CONTINUOUS, ub=gp.GRB.INFINITY, name='u')

    for k in K:
        m_S.addConstr(gp.quicksum(A[k][j] * m_S._x[j] for j in J) == sum(B[i][k] for i in S))
    for i in S:
        m_S.addConstr(m_S._u[i] == gp.quicksum(V[i][j] * m_S._x[j] for j in J))

    m_S.setObjective(m_S._lam)

    try:
        constrs = []
        for i in S:
            constr = m_S.addConstr(m_S._lam <= m_S._u[i] - u_N[i])
            constrs.append(constr)
        m_S.optimize()
        assert m_S._lam.X > epsTh
        m_S.remove(constrs)
        m_S.reset()
    except (AttributeError, AssertionError):
        return None

    min_lam, max_lam = 1, 1
    intersections = []
    for var in m.getVars():

        if var.VBasis == BASIC:
            continue
        if var.VarName in basis_varnames:  # gurobi gimmick because of CBasis
            continue

        row_indices, values = [], []
        col = m.getCol(var)
        for i in range(col.size()):
            coeff, constrname = col.getCoeff(i), col.getConstr(i).ConstrName
            row_indices.append(constr_names_to_indices[constrname])
            values.append(coeff)
        col = ss.csr_matrix((values, (row_indices, np.zeros_like(row_indices))), shape=(m.NumConstrs, 1))
        # inv_basis_mat_col = ss.linalg.spsolve(basis_mat, col)
        inv_basis_mat_col = basis_mat_lu.solve(col.toarray())

        inv_basis_mat_col = np.asarray(inv_basis_mat_col, dtype=float).reshape(-1)

        # r = {v.VarName: 0 for v in m.getVars()}
        r = {
            basis_varname: float(-coeff)
            for coeff, basis_varname in zip(inv_basis_mat_col, basis_varnames)
            if basis_varname[0] == 'u'
            }
        r[var.VarName] = 1.0
        # for i, basis_varname in enumerate(basis_varnames):
        #     r[basis_varname] = -inv_basis_mat_col[i]

        if all(r['u[{0}]'.format(i)] <= 0 for i in S if 'u[{0}]'.format(i) in r):
            continue

        constrs = []
        for i in S:
            key = 'u[{0}]'.format(i)
            if key in r:
                coeff = float(r[key])
                constr = m_S.addConstr(float(u_N[i]) + coeff * m_S._lam <= m_S._u[i])
                constrs.append(constr)
        m_S.optimize()
        if m_S.Status == 2:
            if m_S._lam.X < lamRatTh or m_S._lam.X > (1/lamRatTh if lamRatTh > 0 else np.inf):
                return None
            if m_S._lam.X < min_lam:
                min_lam = m_S._lam.X
            if m_S._lam.X > max_lam:
                max_lam = m_S._lam.X
            if min_lam/max_lam < lamRatTh:
                return None
            intersections.append((var.VarName, m_S._lam.X))
        m_S.remove(constrs)
        m_S.reset()

    return intersections


def get_basis(m, constr_names_to_indices):

    if m.Status != 2:
        raise Exception('m.Status: {0}'.format(m.Status))

    basis_varnames = []
    col_index = 0
    row_indices, col_indices, values = [], [], []

    for var in m.getVars():
        if var.VBasis == BASIC:
            basis_varnames.append(var.VarName)
            col = m.getCol(var)
            for i in range(col.size()):
                coeff, constrname = col.getCoeff(i), col.getConstr(i).ConstrName
                row_indices.append(constr_names_to_indices[constrname])
                col_indices.append(col_index)
                values.append(coeff)
            col_index += 1

    for constr in m.getConstrs():  # gurobi gimmick because of CBasis
        if constr.CBASIS == BASIC:
            row = m.getRow(constr)
            for j in range(row.size()):
                var = row.getVar(j)
                if var.VarName[0] == 's':
                    basis_varnames.append(var.VarName)
                    col = m.getCol(var)
                    for i in range(col.size()):
                        coeff, constrname = col.getCoeff(i), col.getConstr(i).ConstrName
                        row_indices.append(constr_names_to_indices[constrname])
                        col_indices.append(col_index)
                        values.append(coeff)
                    col_index += 1
                    break

    basis_mat = ss.csr_matrix((values, (row_indices, col_indices)), shape=(m.NumConstrs, m.NumConstrs))

    return basis_mat, basis_varnames

