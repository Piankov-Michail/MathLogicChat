from itertools import count
import copy
import re

skolem_counter = count(0)

def new_skolem():
    return f"s{next(skolem_counter)}"

def split_top_level_commas(s):
    parts = []
    current = []
    depth = 0
    for char in s:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append(''.join(current).strip())
    return parts

def split_top_level_op(s, op='∧'):
    parts = []
    current = []
    depth = 0
    for char in s:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == op and depth == 0:
            parts.append(''.join(current).strip())
            current = []
            continue
        current.append(char)
    parts.append(''.join(current).strip())
    return [p for p in parts if p]

def is_negated(literal):
    return literal.startswith("¬") or literal.startswith("~") or literal.startswith("!")

def get_positive_form(literal):
    if is_negated(literal):
        if literal.startswith("¬"):
            return literal[1:].strip()
        elif literal.startswith("~"):
            return literal[1:].strip()
        elif literal.startswith("!"):
            return literal[1:].strip()
    return literal

def negate_predicate(pred_name):
    if pred_name.startswith("not_"):
        return pred_name[4:]
    else:
        return "not_" + pred_name

def parse_atom(atom_str, var_to_const=None):
    atom_str = atom_str.strip()
    var_to_const = var_to_const or {}
    
    pred_match = re.match(r"([a-zA-Z_]\w*)\s*\(([^)]*)\)", atom_str)
    if pred_match:
        pred_name = pred_match.group(1)
        args_str = pred_match.group(2).strip()
        
        if args_str:
            args = [arg.strip() for arg in args_str.split(",")]
            args = [var_to_const.get(arg, arg) for arg in args]
            return (pred_name,) + tuple(args)
        else:
            return (pred_name,)
    
    atom = var_to_const.get(atom_str, atom_str)
    return (atom,)

def process_literal(literal, var_to_const=None, is_antecedent=False):
    literal = literal.strip()
    var_to_const = var_to_const or {}
    
    is_neg = is_negated(literal)
    pos_literal = get_positive_form(literal)
    
    atom_result = parse_atom(pos_literal, var_to_const)
    
    if isinstance(atom_result, tuple):
        pred_name = atom_result[0]
        args = atom_result[1:]
        
        if is_neg:
            return (negate_predicate(pred_name),) + args
        else:
            if is_antecedent:
                return (negate_predicate(pred_name),) + args
            else:
                return (pred_name,) + args
    else:
        atom = atom_result
        if is_neg:
            return ("not_" + atom,)
        else:
            if is_antecedent:
                return ("not_" + atom,)
            else:
                return (atom,)

def process_all_statement(stmt):
    inner = stmt[4:-1].strip()
    parts = split_top_level_commas(inner)
    
    if len(parts) < 2:
        raise ValueError(f"Недостаточно аргументов в all: {stmt}")
    
    var_list = [v.strip() for v in parts[:-1]]
    body = parts[-1].strip()
    
    var_to_upper = {var: var.upper() for var in var_list}
    
    for var, upper in var_to_upper.items():
        var_escaped = re.escape(var)
        body = re.sub(rf'\b{var_escaped}\b', upper, body)
    
    if "→" in body:
        antecedent, consequent = [part.strip() for part in body.split("→", 1)]
        
        if antecedent.startswith("(") and antecedent.endswith(")"):
            antecedent = antecedent[1:-1].strip()
        if consequent.startswith("(") and consequent.endswith(")"):
            consequent = consequent[1:-1].strip()
        
        ant_lits = split_top_level_op(antecedent, '∧') if "∧" in antecedent else [antecedent]
        cons_lits = split_top_level_op(consequent, '∨') if "∨" in consequent else [consequent]
        
        clause_lits = []
        
        for lit_str in ant_lits:
            lit_str = lit_str.strip()
            if not lit_str:
                continue
            
            if is_negated(lit_str):
                pos_lit = get_positive_form(lit_str)
                processed = process_literal(pos_lit, var_to_upper, is_antecedent=False)
                clause_lits.append(processed)
            else:
                processed = process_literal(lit_str, var_to_upper, is_antecedent=True)
                clause_lits.append(processed)
        
        for lit_str in cons_lits:
            lit_str = lit_str.strip()
            if not lit_str:
                continue
            
            if lit_str.startswith("all(") and lit_str.endswith(")"):
                nested_clauses = process_all_statement(lit_str)
                for nested_clause in nested_clauses:
                    clause_lits.extend(list(nested_clause))
            else:
                if is_negated(lit_str):
                    pos_lit = get_positive_form(lit_str)
                    processed = process_literal(pos_lit, var_to_upper, is_antecedent=True)
                    clause_lits.append(processed)
                else:
                    processed = process_literal(lit_str, var_to_upper, is_antecedent=False)
                    clause_lits.append(processed)
        
        return [frozenset(clause_lits)]
    
    else:
        processed = process_literal(body, var_to_upper, is_antecedent=False)
        return [frozenset({processed})]

def process_some_statement(stmt):
    inner = stmt[5:-1].strip()
    parts = split_top_level_commas(inner)
    
    if len(parts) < 2:
        raise ValueError(f"Недостаточно аргументов в some: {stmt}")
    
    var_list = [v.strip() for v in parts[:-1]]
    body = parts[-1].strip()
    
    skolem_consts = [new_skolem() for _ in var_list]
    var_to_const = dict(zip(var_list, skolem_consts))
    
    body_replaced = body
    for var, const in zip(var_list, skolem_consts):
        var_escaped = re.escape(var)
        body_replaced = re.sub(rf'\b{var_escaped}\b', const, body_replaced)
    
    literals = split_top_level_op(body_replaced, '∧')
    clauses = []
    
    for lit_str in literals:
        lit_str = lit_str.strip()
        if not lit_str:
            continue
        
        if lit_str.startswith("all(") and lit_str.endswith(")"):
            nested_clauses = process_all_statement(lit_str)
            clauses.extend(nested_clauses)
        elif lit_str.startswith("some(") and lit_str.endswith(")"):
            raise ValueError("Nested some not supported yet")
        else:
            processed = process_literal(lit_str, {})
            clauses.append(frozenset({processed}))
    
    return clauses

def process_atomic_statement(stmt):
    stmt = stmt.strip()
    
    if is_negated(stmt):
        pos_stmt = get_positive_form(stmt)
        processed = process_literal(pos_stmt, {})
        return [frozenset({processed})]
    else:
        processed = process_literal(stmt, {})
        return [frozenset({processed})]

def parse_statements_and_goal(data):
    global skolem_counter
    skolem_counter = count(0)
    kb_clauses = []
    
    for stmt in data["statements"]:
        stmt = stmt.strip()
        
        if not stmt:
            continue
        
        if stmt.startswith("some(") and stmt.endswith(")"):
            clauses = process_some_statement(stmt)
            kb_clauses.extend(clauses)
        
        elif stmt.startswith("all(") and stmt.endswith(")"):
            clauses = process_all_statement(stmt)
            kb_clauses.extend(clauses)
        
        else:
            try:
                clauses = process_atomic_statement(stmt)
                kb_clauses.extend(clauses)
            except Exception as e:
                raise ValueError(f"Ошибка при обработке атомарного утверждения '{stmt}': {str(e)}")
    
    goal = data["goal"].strip()
    
    if not goal:
        raise ValueError("Цель не может быть пустой")
    
    if goal.startswith("all(") and goal.endswith(")"):
        clauses = process_all_statement(goal)
        for clause in clauses:
            vars_set = set()
            for lit in clause:
                for arg in lit[1:]:
                    if is_variable(arg):
                        vars_set.add(arg)
            var_to_skolem = {v: new_skolem() for v in vars_set}
            
            for lit in clause:
                comp_lit = complement(lit)
                sub_lit = substitute(comp_lit, var_to_skolem)
                kb_clauses.append(frozenset({sub_lit}))
    
    elif is_negated(goal):
        pos_goal = get_positive_form(goal)
        processed = process_literal(pos_goal, {})
        kb_clauses.append(frozenset({processed}))
    
    else:
        processed = process_literal(goal, {})
        if isinstance(processed, tuple):
            negated = (negate_predicate(processed[0]),) + processed[1:]
        else:
            negated = ("not_" + processed,)
        kb_clauses.append(frozenset({negated}))
    
    filtered_clauses = []
    seen = set()
    
    for clause in kb_clauses:
        if contains_complementary_pair(clause):
            continue
        
        clause_tuple = tuple(sorted(clause))
        if clause_tuple in seen:
            continue
        
        seen.add(clause_tuple)
        filtered_clauses.append(clause)
    
    return filtered_clauses

def contains_complementary_pair(clause):
    literals = list(clause)
    for i in range(len(literals)):
        for j in range(i + 1, len(literals)):
            lit1 = literals[i]
            lit2 = literals[j]
            
            if isinstance(lit1, tuple) and isinstance(lit2, tuple):
                name1, name2 = lit1[0], lit2[0]
                args1, args2 = lit1[1:], lit2[1:]
                
                if args1 == args2:
                    if (name1.startswith("not_") and name1[4:] == name2) or \
                       (name2.startswith("not_") and name2[4:] == name1):
                        return True
    return False

def print_clauses(clauses):
    for i, clause in enumerate(clauses, 1):
        literals = []
        for lit in clause:
            if isinstance(lit, tuple):
                pred = lit[0]
                args = lit[1:]
                if args:
                    literals.append(f"{pred}({', '.join(args)})")
                else:
                    literals.append(pred)
            else:
                literals.append(str(lit))
        print(f"{i}: {' ∨ '.join(literals)}")

def is_variable(x):
    return isinstance(x, str) and x[0].isupper()

def unify_var(var, x, theta):
    if var in theta:
        return unify(theta[var], x, theta)
    elif x in theta:
        return unify(var, theta[x], theta)
    else:
        theta[var] = x
        return theta

def unify(x, y, theta=None):
    if theta is None:
        theta = {}
    if theta is None:
        return None
    if x == y:
        return theta
    if is_variable(x):
        return unify_var(x, y, theta)
    if is_variable(y):
        return unify_var(y, x, theta)
    if isinstance(x, tuple) and isinstance(y, tuple):
        if x[0] != y[0] or len(x) != len(y):
            return None
        for i in range(1, len(x)):
            theta = unify(x[i], y[i], theta)
            if theta is None:
                return None
        return theta
    return None

def substitute(literal, theta):
    if is_variable(literal):
        return theta.get(literal, literal)
    if isinstance(literal, tuple):
        return tuple(substitute(arg, theta) for arg in literal)
    return literal

def negate_literal(lit):
    if lit.startswith('not_'):
        return lit[4:]
    else:
        return 'not_' + lit

def complement(lit):
    if isinstance(lit, tuple):
        pred = lit[0]
        args = lit[1:]
        if pred.startswith('not_'):
            return (pred[4:],) + args
        else:
            return ('not_' + pred,) + args
    else:
        if lit.startswith('not_'):
            return lit[4:]
        else:
            return 'not_' + lit

def resolve_clauses(c1, c2):
    results = []
    for l1 in c1:
        for l2 in c2:
            comp_l1 = complement(l1)
            if comp_l1[0] == l2[0] and len(comp_l1) == len(l2):
                theta = unify(comp_l1, l2, {})
                if theta is not None:
                    sub_l1 = substitute(l1, theta)
                    sub_l2 = substitute(l2, theta)

                    new_clause_literals = []
                    for l in c1:
                        sl = substitute(l, theta)
                        if sl != sub_l1:
                            new_clause_literals.append(sl)
                    for l in c2:
                        sl = substitute(l, theta)
                        if sl != sub_l2:
                            new_clause_literals.append(sl)

                    new_clause = frozenset(new_clause_literals)

                    if contains_complementary_pair(new_clause):
                        continue

                    results.append((new_clause, l1, l2, theta))
    return results

def resolution_proof(kb_clauses):
    clause_list = list(kb_clauses)
    filtered = []
    for c in clause_list:
        if not contains_complementary_pair(c):
            filtered.append(c)
        else:
            print(f"Пропущена тавтология: {list(c)}")
    clause_list = filtered

    print("\n=== Начальные клаузы ===")
    for i, c in enumerate(clause_list):
        print(f"{i+1}: {list(c)}")

    step = 1
    n_start = len(clause_list)

    while True:
        n = len(clause_list)
        new_resolvents = []
        found_new = False

        for i in range(n):
            for j in range(i, n):
                c1 = clause_list[i]
                c2 = clause_list[j]
                resolvents_info = resolve_clauses(c1, c2)
                for res, lit1, lit2, theta in resolvents_info:
                    if len(res) == 0:
                        print(f"\nШаг {step}: Резолюция между клаузами {i+1} и {j+1}")
                        print(f"    Клауза {i+1}: {list(c1)}")
                        print(f"    Клауза {j+1}: {list(c2)}")
                        print(f"    Резольвируемые литералы: {lit1} и {lit2}")
                        print(f"    Подстановка: {theta}")
                        print(f"    ➤ Получена пустая клауза! Противоречие найдено.")
                        return True
                    if res not in clause_list:
                        new_resolvents.append((res, i, j, lit1, lit2, theta))
                        found_new = True

        if not found_new:
            print("\nНовых клауз больше нет. Противоречие не найдено.")
            return False

        print(f"\n=== Шаг {step} ===")
        added_any = False
        for res, i, j, lit1, lit2, theta in new_resolvents:
            if res in clause_list:
                continue
            clause_list.append(res)
            idx = len(clause_list)
            print(f"Резолюция {i+1} & {j+1} → клауза {idx}")
            print(f"  {list(clause_list[i])}")
            print(f"  {list(clause_list[j])}")
            print(f"  Литералы: {lit1} и {lit2}")
            print(f"  Подстановка: {theta}")
            print(f"  ➤ {list(res)}")
            added_any = True

        if not added_any:
            print("Нет новых клауз для добавления.")
            break

        step += 1
        if step > 50:
            print("Достигнут лимит шагов (50). Прекращаем.")
            break

    print("\nДоказательство не удалось.")
    return False

import io
import sys
from contextlib import redirect_stdout

def solve_logic_task(data: dict) -> str:
    try:
        kb = parse_statements_and_goal(data)
        
        f = io.StringIO()
        with redirect_stdout(f):
            success = resolution_proof(kb)
            result = "Доказательство успешно!" if success else "Доказательство не удалось."
        
        output = f.getvalue()
        return str(data) + output + "\n" + result
    except Exception as e:
        return f"Ошибка при решении задачи:\n{str(e)}"

if __name__ == "__main__":

    data = {
        "statements": [
            "some(p, patient(p) ∧ all(d, doctor(d) → loves(p, d)))",
            "all(p, z, (patient(p) ∧ healer(z)) → ¬loves(p, z))"
        ],
        "goal": "all(d, z, (doctor(d) ∧ healer(z)) → d ≠ z)"
    }

    kb = parse_statements_and_goal(data)
    print("Сгенерированные клаузы:")
    for c in kb:
        print(c)

    success = resolution_proof(kb)
    print("Доказательство успешно!" if success else "Доказательство не удалось.")