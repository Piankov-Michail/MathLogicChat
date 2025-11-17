import re
from typing import List, Dict, Any, Union, Optional, Set, Tuple, FrozenSet
from collections import defaultdict, deque
import copy
import json
import io
import sys


class Node:
    def __str__(self):
        return self.to_string()
    
    def to_string(self) -> str:
        raise NotImplementedError
    
    def clone(self) -> 'Node':
        raise NotImplementedError
    
    def apply_transformations(self) -> 'Node':
        # Шаг 1: Удалить эквивалентности
        result = self.eliminate_equivalence()
        # Шаг 2: Удалить импликации
        result = result.eliminate_implication()
        # Шаги 3-6: Переместить отрицания внутрь
        result = result.move_negation_inward()
        # Шаг 7: Вынести кванторы влево
        result = result.pull_quantifiers_left()
        # Убрано: convert_to_cnf перемещено после сколемизации
        return result
    
    def eliminate_equivalence(self) -> 'Node':
        raise NotImplementedError
    
    def eliminate_implication(self) -> 'Node':
        raise NotImplementedError
    
    def move_negation_inward(self) -> 'Node':
        raise NotImplementedError
    
    def pull_quantifiers_left(self) -> 'Node':
        raise NotImplementedError
    
    def convert_to_cnf(self) -> 'Node':
        raise NotImplementedError
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        if universal_vars is None:
            universal_vars = []
        if skolem_counter is None:
            skolem_counter = {'count': 0}
        raise NotImplementedError
    
    def remove_quantifiers(self) -> 'Node':
        raise NotImplementedError
    
    def to_cnf_clauses(self) -> List['Clause']:
        raise NotImplementedError


class Term:
    def __str__(self):
        raise NotImplementedError
    
    def clone(self) -> 'Term':
        raise NotImplementedError
    
    def contains_variable(self, var_name: str) -> bool:
        raise NotImplementedError
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'Term':
        raise NotImplementedError
    
    def get_name(self) -> str:
        return str(self)


class Variable(Term):
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def clone(self) -> 'Variable':
        return Variable(self.name)
    
    def contains_variable(self, var_name: str) -> bool:
        return self.name == var_name
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'Term':
        return substitution.get(self.name, self)
    
    def get_name(self) -> str:
        return self.name


class Constant(Term):
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def clone(self) -> 'Constant':
        return Constant(self.name)
    
    def contains_variable(self, var_name: str) -> bool:
        return False
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'Term':
        return self
    
    def get_name(self) -> str:
        return self.name


class Function(Term):
    def __init__(self, name: str, args: List[Term]):
        self.name = name
        self.args = args
    
    def __str__(self):
        if self.args:
            args_str = ', '.join(str(arg) for arg in self.args)
            return f"{self.name}({args_str})"
        return self.name
    
    def clone(self) -> 'Function':
        return Function(self.name, [arg.clone() for arg in self.args])
    
    def contains_variable(self, var_name: str) -> bool:
        return any(arg.contains_variable(var_name) for arg in self.args)
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'Term':
        return Function(self.name, [arg.substitute(substitution) for arg in self.args])
    
    def get_name(self) -> str:
        return str(self)


class Unifier:
    
    @staticmethod
    def unify(term1: Term, term2: Term, substitution: Dict[str, Term] = None) -> Optional[Dict[str, Term]]:
        if substitution is None:
            substitution = {}
        
        term1 = term1.substitute(substitution)
        term2 = term2.substitute(substitution)
        
        if str(term1) == str(term2):
            return substitution
        
        if isinstance(term1, Variable):
            return Unifier._unify_variable(term1, term2, substitution)
        
        if isinstance(term2, Variable):
            return Unifier._unify_variable(term2, term1, substitution)
        
        if isinstance(term1, Function) and isinstance(term2, Function):
            if term1.name != term2.name or len(term1.args) != len(term2.args):
                return None
            
            current_subst = substitution.copy()
            for arg1, arg2 in zip(term1.args, term2.args):
                result = Unifier.unify(arg1, arg2, current_subst)
                if result is None:
                    return None
                current_subst = result
            
            return current_subst
        
        return None
    
    @staticmethod
    def _unify_variable(var: Variable, term: Term, substitution: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        var_name = var.name
        
        if var_name in substitution:
            return Unifier.unify(substitution[var_name], term, substitution)
        
        if term.contains_variable(var_name):
            return None
        
        new_substitution = substitution.copy()
        new_substitution[var_name] = term
        return new_substitution


class Literal:
    def __init__(self, predicate: str, args: List[Term], negated: bool = False):
        self.predicate = predicate
        self.args = args
        self.negated = negated
    
    def __str__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        pred_str = f"{self.predicate}({args_str})" if self.args else self.predicate
        return f"¬{pred_str}" if self.negated else pred_str
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return (self.predicate == other.predicate and 
                all(str(a1) == str(a2) for a1, a2 in zip(self.args, other.args)) and 
                self.negated == other.negated)
    
    def __hash__(self):
        return hash((self.predicate, tuple(str(arg) for arg in self.args), self.negated))
    
    def negate(self) -> 'Literal':
        return Literal(self.predicate, self.args.copy(), not self.negated)
    
    def is_complementary(self, other: 'Literal') -> bool:
        if self.predicate != other.predicate or self.negated == other.negated:
            return False
        return all(str(a1) == str(a2) for a1, a2 in zip(self.args, other.args))
    
    def most_general_unifier(self, other: 'Literal') -> Optional[Dict[str, Term]]:
        if self.predicate != other.predicate or len(self.args) != len(other.args):
            return None
        
        substitution = {}
        for arg1, arg2 in zip(self.args, other.args):
            result = Unifier.unify(arg1, arg2, substitution)
            if result is None:
                return None
            substitution = result
        
        return substitution
    
    def substitute(self, substitution: Dict[str, Term]) -> 'Literal':
        new_args = [arg.substitute(substitution) for arg in self.args]
        return Literal(self.predicate, new_args, self.negated)


class Clause:
    def __init__(self, literals: Set[Literal]):
        self.literals = literals
    
    def __str__(self):
        if not self.literals:
            return "□"
        return " ∨ ".join(str(lit) for lit in sorted(self.literals, key=lambda x: str(x)))
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if not isinstance(other, Clause):
            return False
        return self.literals == other.literals
    
    def __hash__(self):
        return hash(frozenset(str(lit) for lit in self.literals))
    
    def is_empty(self) -> bool:
        return len(self.literals) == 0
    
    def is_tautology(self) -> bool:
        for lit1 in self.literals:
            for lit2 in self.literals:
                if lit1.predicate == lit2.predicate and lit1.negated != lit2.negated:
                    if all(str(a1) == str(a2) for a1, a2 in zip(lit1.args, lit2.args)):
                        return True
        return False
    
    def resolve_with(self, other: 'Clause') -> List[Tuple['Clause', Dict[str, Term]]]:
        resolvents = []
        
        for lit1 in self.literals:
            for lit2 in other.literals:
                if lit1.predicate == lit2.predicate and lit1.negated != lit2.negated:
                    mgu = lit1.most_general_unifier(lit2)
                    if mgu is not None:
                        new_literals1 = {l.substitute(mgu) for l in self.literals - {lit1}}
                        new_literals2 = {l.substitute(mgu) for l in other.literals - {lit2}}
                        
                        new_literals = new_literals1.union(new_literals2)
                        
                        new_clause = Clause(new_literals)
                        
                        if not new_clause.is_tautology():
                            resolvents.append((new_clause, mgu))
                        
                        if new_clause.is_empty():
                            resolvents.append((new_clause, mgu))
        
        return resolvents
    
    def get_variables(self) -> Set[str]:
        variables = set()
        for lit in self.literals:
            for arg in lit.args:
                if isinstance(arg, Variable):
                    variables.add(arg.name)
        return variables


class VariableNode(Node):
    def __init__(self, name: str):
        self.name = name
    
    def to_string(self) -> str:
        return self.name
    
    def clone(self) -> 'VariableNode':
        return VariableNode(self.name)
    
    def eliminate_equivalence(self) -> 'Node':
        return self.clone()
    
    def eliminate_implication(self) -> 'Node':
        return self.clone()
    
    def move_negation_inward(self) -> 'Node':
        return self.clone()
    
    def pull_quantifiers_left(self) -> 'Node':
        return self.clone()
    
    def convert_to_cnf(self) -> 'Node':
        return self.clone()
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return self.clone()
    
    def remove_quantifiers(self) -> 'Node':
        return self.clone()
    
    def to_cnf_clauses(self) -> List['Clause']:
        raise ValueError("Переменная не может быть преобразована в клаузы КНФ")


class ConstantNode(Node):
    def __init__(self, name: str):
        self.name = name
    
    def to_string(self) -> str:
        return self.name
    
    def clone(self) -> 'ConstantNode':
        return ConstantNode(self.name)
    
    def eliminate_equivalence(self) -> 'Node':
        return self.clone()
    
    def eliminate_implication(self) -> 'Node':
        return self.clone()
    
    def move_negation_inward(self) -> 'Node':
        return self.clone()
    
    def pull_quantifiers_left(self) -> 'Node':
        return self.clone()
    
    def convert_to_cnf(self) -> 'Node':
        return self.clone()
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return self.clone()
    
    def remove_quantifiers(self) -> 'Node':
        return self.clone()
    
    def to_cnf_clauses(self) -> List['Clause']:
        raise ValueError("Константа не может быть преобразована в клаузы КНФ")


class SkolemFunctionNode(Node):
    def __init__(self, base_name: str, args: List[Node]):
        self.base_name = base_name
        self.args = args
        self.name = self._generate_name()
    
    def _generate_name(self) -> str:
        return f"sk{len(self.args)}"
    
    def to_string(self) -> str:
        if self.args:
            args_str = ', '.join(str(arg) for arg in self.args)
            return f"{self.name}({args_str})"
        return self.name
    
    def clone(self) -> 'SkolemFunctionNode':
        return SkolemFunctionNode(self.base_name, [arg.clone() for arg in self.args])
    
    def eliminate_equivalence(self) -> 'Node':
        return SkolemFunctionNode(self.base_name, [arg.eliminate_equivalence() for arg in self.args])
    
    def eliminate_implication(self) -> 'Node':
        return SkolemFunctionNode(self.base_name, [arg.eliminate_implication() for arg in self.args])
    
    def move_negation_inward(self) -> 'Node':
        return SkolemFunctionNode(self.base_name, [arg.move_negation_inward() for arg in self.args])
    
    def pull_quantifiers_left(self) -> 'Node':
        return SkolemFunctionNode(self.base_name, [arg.pull_quantifiers_left() for arg in self.args])
    
    def convert_to_cnf(self) -> 'Node':
        return SkolemFunctionNode(self.base_name, [arg.convert_to_cnf() for arg in self.args])
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return SkolemFunctionNode(self.base_name, [arg.skolemize(universal_vars, skolem_counter) for arg in self.args])
    
    def remove_quantifiers(self) -> 'Node':
        return SkolemFunctionNode(self.base_name, [arg.remove_quantifiers() for arg in self.args])
    
    def to_cnf_clauses(self) -> List['Clause']:
        raise ValueError("Функция Сколема не может быть преобразована в клаузы КНФ")


class PredicateNode(Node):
    def __init__(self, name: str, args: List[Node]):
        self.name = name
        self.args = args
    
    def to_string(self) -> str:
        if self.args:
            args_str = ', '.join(str(arg) for arg in self.args)
            return f"{self.name}({args_str})"
        return self.name
    
    def clone(self) -> 'PredicateNode':
        return PredicateNode(self.name, [arg.clone() for arg in self.args])
    
    def eliminate_equivalence(self) -> 'Node':
        return PredicateNode(self.name, [arg.eliminate_equivalence() for arg in self.args])
    
    def eliminate_implication(self) -> 'Node':
        return PredicateNode(self.name, [arg.eliminate_implication() for arg in self.args])
    
    def move_negation_inward(self) -> 'Node':
        return PredicateNode(self.name, [arg.move_negation_inward() for arg in self.args])
    
    def pull_quantifiers_left(self) -> 'Node':
        return PredicateNode(self.name, [arg.pull_quantifiers_left() for arg in self.args])
    
    def convert_to_cnf(self) -> 'Node':
        return PredicateNode(self.name, [arg.convert_to_cnf() for arg in self.args])
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return PredicateNode(self.name, [arg.skolemize(universal_vars, skolem_counter) for arg in self.args])
    
    def remove_quantifiers(self) -> 'Node':
        return PredicateNode(self.name, [arg.remove_quantifiers() for arg in self.args])
    
    def to_cnf_clauses(self) -> List['Clause']:
        args = []
        for arg in self.args:
            if isinstance(arg, VariableNode):
                args.append(Variable(arg.name))
            elif isinstance(arg, ConstantNode):
                args.append(Constant(arg.name))
            elif isinstance(arg, SkolemFunctionNode):
                args.append(Constant(str(arg)))
            else:
                args.append(Constant(str(arg)))
        
        literal = Literal(self.name, args, negated=False)
        return [Clause({literal})]


class NotNode(Node):
    def __init__(self, child: Node):
        self.child = child
    
    def to_string(self) -> str:
        if isinstance(self.child, (AndNode, OrNode, ImpliesNode, EquivNode, ForallNode, ExistsNode)):
            return f"¬({self.child})"
        return f"¬{self.child}"
    
    def clone(self) -> 'NotNode':
        return NotNode(self.child.clone())
    
    def eliminate_equivalence(self) -> 'Node':
        child_transformed = self.child.eliminate_equivalence()
        
        if isinstance(child_transformed, EquivNode):
            a = child_transformed.left.eliminate_equivalence()
            b = child_transformed.right.eliminate_equivalence()
            return OrNode(
                AndNode(a.clone(), NotNode(b.clone())),
                AndNode(NotNode(a.clone()), b.clone())
            )
        
        return NotNode(child_transformed)
    
    def eliminate_implication(self) -> 'Node':
        child_transformed = self.child.eliminate_implication()
        
        if isinstance(child_transformed, ImpliesNode):
            a = child_transformed.left.eliminate_implication()
            b = child_transformed.right.eliminate_implication()
            return AndNode(a.clone(), NotNode(b.clone()))
        
        return NotNode(child_transformed)
    
    def move_negation_inward(self) -> 'Node':
        child = self.child.move_negation_inward()
        
        if isinstance(child, ForallNode):
            return ExistsNode(child.var_name, NotNode(child.body).move_negation_inward())
        
        if isinstance(child, ExistsNode):
            return ForallNode(child.var_name, NotNode(child.body).move_negation_inward())
        
        if isinstance(child, AndNode):
            return OrNode(
                NotNode(child.left).move_negation_inward(),
                NotNode(child.right).move_negation_inward()
            )
        
        if isinstance(child, OrNode):
            return AndNode(
                NotNode(child.left).move_negation_inward(),
                NotNode(child.right).move_negation_inward()
            )
        
        if isinstance(child, (PredicateNode, VariableNode, ConstantNode)):
            return NotNode(child)
        
        if isinstance(child, NotNode):
            return child.child.move_negation_inward()
        
        return NotNode(child)
    
    def pull_quantifiers_left(self) -> 'Node':
        child = self.child.pull_quantifiers_left()
        return NotNode(child)
    
    def convert_to_cnf(self) -> 'Node':
        child = self.child.convert_to_cnf()
        
        if isinstance(child, AndNode):
            return OrNode(
                NotNode(child.left).convert_to_cnf(),
                NotNode(child.right).convert_to_cnf()
            )
        
        if isinstance(child, OrNode):
            return AndNode(
                NotNode(child.left).convert_to_cnf(),
                NotNode(child.right).convert_to_cnf()
            )
        
        return NotNode(child)
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return NotNode(self.child.skolemize(universal_vars, skolem_counter))
    
    def remove_quantifiers(self) -> 'Node':
        return NotNode(self.child.remove_quantifiers())
    
    def to_cnf_clauses(self) -> List['Clause']:
        if isinstance(self.child, PredicateNode):
            args = []
            for arg in self.child.args:
                if isinstance(arg, VariableNode):
                    args.append(Variable(arg.name))
                elif isinstance(arg, ConstantNode):
                    args.append(Constant(arg.name))
                elif isinstance(arg, SkolemFunctionNode):
                    args.append(Constant(str(arg)))
                else:
                    args.append(Constant(str(arg)))
            
            literal = Literal(self.child.name, args, negated=True)
            return [Clause({literal})]
        raise ValueError("Отрицание может быть применено только к предикату в КНФ")


class AndNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right
    
    def to_string(self) -> str:
        left_str = str(self.left)
        right_str = str(self.right)
        if isinstance(self.left, OrNode):
            left_str = f"({left_str})"
        if isinstance(self.right, OrNode):
            right_str = f"({right_str})"
        return f"{left_str} ∧ {right_str}"
    
    def clone(self) -> 'AndNode':
        return AndNode(self.left.clone(), self.right.clone())
    
    def eliminate_equivalence(self) -> 'Node':
        return AndNode(
            self.left.eliminate_equivalence(),
            self.right.eliminate_equivalence()
        )
    
    def eliminate_implication(self) -> 'Node':
        return AndNode(
            self.left.eliminate_implication(),
            self.right.eliminate_implication()
        )
    
    def move_negation_inward(self) -> 'Node':
        return AndNode(
            self.left.move_negation_inward(),
            self.right.move_negation_inward()
        )
    
    def pull_quantifiers_left(self) -> 'Node':
        left = self.left.pull_quantifiers_left()
        right = self.right.pull_quantifiers_left()
        return AndNode(left, right)
    
    def convert_to_cnf(self) -> 'Node':
        left = self.left.convert_to_cnf()
        right = self.right.convert_to_cnf()
        
        if isinstance(left, OrNode):
            return AndNode(left, right)
        
        if isinstance(right, OrNode):
            return AndNode(left, right)
        
        return AndNode(left, right)
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return AndNode(
            self.left.skolemize(universal_vars, skolem_counter),
            self.right.skolemize(universal_vars, skolem_counter)
        )
    
    def remove_quantifiers(self) -> 'Node':
        return AndNode(
            self.left.remove_quantifiers(),
            self.right.remove_quantifiers()
        )
    
    def to_cnf_clauses(self) -> List['Clause']:
        left_clauses = self.left.to_cnf_clauses()
        right_clauses = self.right.to_cnf_clauses()
        return left_clauses + right_clauses


class OrNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right
    
    def to_string(self) -> str:
        return f"{self.left} ∨ {self.right}"
    
    def clone(self) -> 'OrNode':
        return OrNode(self.left.clone(), self.right.clone())
    
    def eliminate_equivalence(self) -> 'Node':
        return OrNode(
            self.left.eliminate_equivalence(),
            self.right.eliminate_equivalence()
        )
    
    def eliminate_implication(self) -> 'Node':
        return OrNode(
            self.left.eliminate_implication(),
            self.right.eliminate_implication()
        )
    
    def move_negation_inward(self) -> 'Node':
        return OrNode(
            self.left.move_negation_inward(),
            self.right.move_negation_inward()
        )
    
    def pull_quantifiers_left(self) -> 'Node':
        left = self.left.pull_quantifiers_left()
        right = self.right.pull_quantifiers_left()
        return OrNode(left, right)
    
    def convert_to_cnf(self) -> 'Node':
        left = self.left.convert_to_cnf()
        right = self.right.convert_to_cnf()
        
        if isinstance(left, AndNode):
            return AndNode(
                OrNode(left.left.clone(), right.clone()).convert_to_cnf(),
                OrNode(left.right.clone(), right.clone()).convert_to_cnf()
            )
        
        if isinstance(right, AndNode):
            return AndNode(
                OrNode(left.clone(), right.left.clone()).convert_to_cnf(),
                OrNode(left.clone(), right.right.clone()).convert_to_cnf()
            )
        
        return OrNode(left, right)
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return OrNode(
            self.left.skolemize(universal_vars, skolem_counter),
            self.right.skolemize(universal_vars, skolem_counter)
        )
    
    def remove_quantifiers(self) -> 'Node':
        return OrNode(
            self.left.remove_quantifiers(),
            self.right.remove_quantifiers()
        )
    
    def to_cnf_clauses(self) -> List['Clause']:
        left_literals = self._extract_literals(self.left)
        right_literals = self._extract_literals(self.right)
        
        all_literals = left_literals.union(right_literals)
        
        for lit1 in all_literals:
            for lit2 in all_literals:
                if lit1.predicate == lit2.predicate and lit1.negated != lit2.negated:
                    if all(str(a1) == str(a2) for a1, a2 in zip(lit1.args, lit2.args)):
                        return []
        
        return [Clause(all_literals)]
    
    def _extract_literals(self, node: Node) -> Set[Literal]:
        if isinstance(node, PredicateNode):
            args = []
            for arg in node.args:
                if isinstance(arg, VariableNode):
                    args.append(Variable(arg.name))
                elif isinstance(arg, ConstantNode):
                    args.append(Constant(arg.name))
                else:
                    args.append(Constant(str(arg)))
            return {Literal(node.name, args, negated=False)}
        
        if isinstance(node, NotNode) and isinstance(node.child, PredicateNode):
            args = []
            for arg in node.child.args:
                if isinstance(arg, VariableNode):
                    args.append(Variable(arg.name))
                elif isinstance(arg, ConstantNode):
                    args.append(Constant(arg.name))
                else:
                    args.append(Constant(str(arg)))
            return {Literal(node.child.name, args, negated=True)}
        
        if isinstance(node, OrNode):
            left_lits = self._extract_literals(node.left)
            right_lits = self._extract_literals(node.right)
            return left_lits.union(right_lits)
        
        if isinstance(node, AndNode):
            raise ValueError("Некорректная КНФ: дизъюнкция содержит конъюнкцию")
        
        raise ValueError(f"Неизвестный тип узла в КНФ: {type(node)}")


class ImpliesNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right
    
    def to_string(self) -> str:
        return f"{self.left} → {self.right}"
    
    def clone(self) -> 'ImpliesNode':
        return ImpliesNode(self.left.clone(), self.right.clone())
    
    def eliminate_equivalence(self) -> 'Node':
        left = self.left.eliminate_equivalence()
        right = self.right.eliminate_equivalence()
        return ImpliesNode(left, right)
    
    def eliminate_implication(self) -> 'Node':
        left = self.left.eliminate_implication()
        right = self.right.eliminate_implication()
        return OrNode(NotNode(left), right)
    
    def move_negation_inward(self) -> 'Node':
        return self.clone()
    
    def pull_quantifiers_left(self) -> 'Node':
        return self.clone()
    
    def convert_to_cnf(self) -> 'Node':
        return self.clone()
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return self.clone()
    
    def remove_quantifiers(self) -> 'Node':
        return self.clone()
    
    def to_cnf_clauses(self) -> List['Clause']:
        return self.eliminate_implication().to_cnf_clauses()


class EquivNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right
    
    def to_string(self) -> str:
        return f"{self.left} ↔ {self.right}"
    
    def clone(self) -> 'EquivNode':
        return EquivNode(self.left.clone(), self.right.clone())
    
    def eliminate_equivalence(self) -> 'Node':
        left = self.left.eliminate_equivalence()
        right = self.right.eliminate_equivalence()
        return AndNode(
            OrNode(NotNode(left.clone()), right.clone()),
            OrNode(left.clone(), NotNode(right.clone()))
        )
    
    def eliminate_implication(self) -> 'Node':
        return self.clone()
    
    def move_negation_inward(self) -> 'Node':
        return self.clone()
    
    def pull_quantifiers_left(self) -> 'Node':
        return self.clone()
    
    def convert_to_cnf(self) -> 'Node':
        return self.clone()
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        return self.clone()
    
    def remove_quantifiers(self) -> 'Node':
        return self.clone()
    
    def to_cnf_clauses(self) -> List['Clause']:
        return self.eliminate_equivalence().to_cnf_clauses()


class ForallNode(Node):
    def __init__(self, var_name: str, body: Node):
        self.var_name = var_name
        self.body = body
    
    def to_string(self) -> str:
        return f"∀{self.var_name}({self.body})"
    
    def clone(self) -> 'ForallNode':
        return ForallNode(self.var_name, self.body.clone())
    
    def eliminate_equivalence(self) -> 'Node':
        return ForallNode(self.var_name, self.body.eliminate_equivalence())
    
    def eliminate_implication(self) -> 'Node':
        return ForallNode(self.var_name, self.body.eliminate_implication())
    
    def move_negation_inward(self) -> 'Node':
        return ForallNode(self.var_name, self.body.move_negation_inward())
    
    def pull_quantifiers_left(self) -> 'Node':
        body = self.body.pull_quantifiers_left()
        return ForallNode(self.var_name, body)
    
    def convert_to_cnf(self) -> 'Node':
        return ForallNode(self.var_name, self.body.convert_to_cnf())
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        if universal_vars is None:
            universal_vars = []
        if skolem_counter is None:
            skolem_counter = {'count': 0}
        
        new_universal_vars = universal_vars + [self.var_name]
        skolemized_body = self.body.skolemize(new_universal_vars, skolem_counter)
        return skolemized_body
    
    def remove_quantifiers(self) -> 'Node':
        return self.body.remove_quantifiers()
    
    def to_cnf_clauses(self) -> List['Clause']:
        return self.body.to_cnf_clauses()


class ExistsNode(Node):
    def __init__(self, var_name: str, body: Node):
        self.var_name = var_name
        self.body = body
    
    def to_string(self) -> str:
        return f"∃{self.var_name}({self.body})"
    
    def clone(self) -> 'ExistsNode':
        return ExistsNode(self.var_name, self.body.clone())
    
    def eliminate_equivalence(self) -> 'Node':
        return ExistsNode(self.var_name, self.body.eliminate_equivalence())
    
    def eliminate_implication(self) -> 'Node':
        return ExistsNode(self.var_name, self.body.eliminate_implication())
    
    def move_negation_inward(self) -> 'Node':
        return ExistsNode(self.var_name, self.body.move_negation_inward())
    
    def pull_quantifiers_left(self) -> 'Node':
        body = self.body.pull_quantifiers_left()
        return ExistsNode(self.var_name, body)
    
    def convert_to_cnf(self) -> 'Node':
        return ExistsNode(self.var_name, self.body.convert_to_cnf())
    
    def skolemize(self, universal_vars: List[str] = None, skolem_counter: Dict[str, int] = None) -> 'Node':
        if universal_vars is None:
            universal_vars = []
        if skolem_counter is None:
            skolem_counter = {'count': 0}
        
        skolem_counter['count'] += 1
        skolem_id = skolem_counter['count']
        
        if not universal_vars:
            skolem_constant = ConstantNode(f"C{skolem_id}")  # Изменено на "C" (заглавная)
            replaced_body = self._replace_variable(self.body, self.var_name, skolem_constant)
        else:
            universal_var_nodes = [VariableNode(var) for var in universal_vars]
            skolem_function = SkolemFunctionNode("f", universal_var_nodes)
            replaced_body = self._replace_variable(self.body, self.var_name, skolem_function)
        
        return replaced_body.skolemize(universal_vars, skolem_counter)
    
    def _replace_variable(self, node: Node, var_name: str, replacement: Node) -> Node:
        if isinstance(node, VariableNode):
            if node.name == var_name:
                return replacement.clone()
            return node.clone()
        
        if isinstance(node, PredicateNode):
            new_args = [self._replace_variable(arg, var_name, replacement) for arg in node.args]
            return PredicateNode(node.name, new_args)
        
        if isinstance(node, NotNode):
            new_child = self._replace_variable(node.child, var_name, replacement)
            return NotNode(new_child)
        
        if isinstance(node, AndNode):
            new_left = self._replace_variable(node.left, var_name, replacement)
            new_right = self._replace_variable(node.right, var_name, replacement)
            return AndNode(new_left, new_right)
        
        if isinstance(node, OrNode):
            new_left = self._replace_variable(node.left, var_name, replacement)
            new_right = self._replace_variable(node.right, var_name, replacement)
            return OrNode(new_left, new_right)
        
        if isinstance(node, ForallNode):
            if node.var_name == var_name:
                return node.clone()
            new_body = self._replace_variable(node.body, var_name, replacement)
            return ForallNode(node.var_name, new_body)
        
        if isinstance(node, ExistsNode):
            if node.var_name == var_name:
                return node.clone()
            new_body = self._replace_variable(node.body, var_name, replacement)
            return ExistsNode(node.var_name, new_body)
        
        if isinstance(node, ConstantNode) or isinstance(node, SkolemFunctionNode):
            return node.clone()
        
        return node.clone()
    
    def remove_quantifiers(self) -> 'Node':
        return self.body.remove_quantifiers()
    
    def to_cnf_clauses(self) -> List['Clause']:
        return self.body.to_cnf_clauses()


class Parser:
    
    @staticmethod
    def parse(formula: str) -> Node:
        formula = formula.strip()
        if not formula:
            raise ValueError("Пустая формула")

        if formula.startswith('∀(') or formula.startswith('∃('):
            return Parser._parse_quantifier_with_parentheses(formula)

        if re.match(r'^[∀∃][a-zA-Z]+', formula):
            return Parser._parse_quantifier(formula)

        return Parser._parse_implication(formula)
    
    @staticmethod
    def _parse_quantifier_with_parentheses(formula: str) -> Node:
        quantifier = formula[0]
        level = 0
        var_part_end = -1

        for i in range(1, len(formula)):
            if formula[i] == '(':
                level += 1
            elif formula[i] == ')':
                level -= 1
                if level == 0 and i < len(formula) - 1:
                    level += 1
            elif formula[i] == ',' and level == 1:
                var_part_end = i
                break

        if var_part_end == -1:
            raise ValueError(f"Не найден разделитель переменных и тела в кванторе: {formula}")

        vars_str = formula[formula.find('(')+1:var_part_end].strip()
        body_str = formula[var_part_end+1 : -1].strip()

        var_names = [v.strip() for v in vars_str.split(',') if v.strip()]

        body_node = Parser.parse(body_str)

        node = body_node
        for var in reversed(var_names):
            if quantifier == '∀':
                node = ForallNode(var, node)
            elif quantifier == '∃':
                node = ExistsNode(var, node)

        return node
    
    @staticmethod
    def _parse_quantifier(formula: str) -> Node:
        quantifier = formula[0]
        var_end = 1
        while var_end < len(formula) and formula[var_end].isalnum():
            var_end += 1
        
        if var_end == 1:
            raise ValueError(f"Некорректный квантор: {formula}")
        
        var_name = formula[1:var_end].strip()
        body = formula[var_end:].strip()
        
        if body.startswith('(') and body.endswith(')'):
            body = body[1:-1].strip()
        
        body_node = Parser.parse(body)
        
        if quantifier == '∀':
            return ForallNode(var_name, body_node)
        elif quantifier == '∃':
            return ExistsNode(var_name, body_node)
        else:
            raise ValueError(f"Неизвестный квантор: {quantifier}")
    
    @staticmethod
    def _parse_implication(formula: str) -> Node:
        level = 0
        i = 0
        while i < len(formula):
            char = formula[i]
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            
            if level == 0:
                if i + 1 <= len(formula) and i > 0:
                    if formula[i] == '→':
                        left = formula[:i].strip()
                        right = formula[i+1:].strip()
                        return ImpliesNode(Parser.parse(left), Parser.parse(right))
                    if formula[i] == '↔':
                        left = formula[:i].strip()
                        right = formula[i+1:].strip()
                        return EquivNode(Parser.parse(left), Parser.parse(right))
            i += 1
        
        return Parser._parse_or(formula)
    
    @staticmethod
    def _parse_or(formula: str) -> Node:
        level = 0
        i = 0
        while i < len(formula):
            char = formula[i]
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            
            if level == 0 and i > 0:
                if formula[i] == '∨':
                    left = formula[:i].strip()
                    right = formula[i+1:].strip()
                    return OrNode(Parser.parse(left), Parser.parse(right))
            i += 1
        
        return Parser._parse_and(formula)
    
    @staticmethod
    def _parse_and(formula: str) -> Node:
        level = 0
        i = 0
        while i < len(formula):
            char = formula[i]
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            
            if level == 0 and i > 0:
                if formula[i] == '∧':
                    left = formula[:i].strip()
                    right = formula[i+1:].strip()
                    return AndNode(Parser.parse(left), Parser.parse(right))
            i += 1
        
        return Parser._parse_not(formula)
    
    @staticmethod
    def _parse_not(formula: str) -> Node:
        formula = formula.strip()
        if formula.startswith('¬'):
            body = formula[1:].strip()
            if body.startswith('(') and body.endswith(')'):
                body = body[1:-1].strip()
            return NotNode(Parser.parse(body))
        
        return Parser._parse_atomic(formula)
    
    @staticmethod
    def _parse_atomic(formula: str) -> Node:
        formula = formula.strip()
        if formula.startswith('(') and formula.endswith(')'):
            return Parser.parse(formula[1:-1].strip())
        
        if '(' in formula and formula.endswith(')'):
            pred_end = formula.find('(')
            pred_name = formula[:pred_end].strip()
            args_str = formula[pred_end+1:-1].strip()
            
            args = []
            current_arg = ""
            level = 0
            for char in args_str:
                if char == '(':
                    level += 1
                elif char == ')':
                    level -= 1
                elif char == ',' and level == 0:
                    if current_arg.strip():
                        args.append(Parser.parse(current_arg.strip()))
                    current_arg = ""
                    continue
                current_arg += char
            
            if current_arg.strip():
                args.append(Parser.parse(current_arg.strip()))
            
            return PredicateNode(pred_name, args)
        
        if len(formula) == 1 and formula.islower() and formula.isalpha():
            return VariableNode(formula)
        else:
            return ConstantNode(formula)


def convert_formula_to_pnf(formula: str) -> str:
    try:
        tree = Parser.parse(formula)
        pnf_tree = tree.apply_transformations()
        return str(pnf_tree)
    except Exception as e:
        print(f"Ошибка при обработке формулы '{formula}': {str(e)}")
        return formula


def convert_formula_to_skolem(formula: str) -> str:
    try:
        tree = Parser.parse(formula)
        transformed_tree = tree.apply_transformations()
        skolem_tree = transformed_tree.skolemize()
        return str(skolem_tree)
    except Exception as e:
        print(f"Ошибка при сколемизации формулы '{formula}': {str(e)}")
        return formula


def convert_to_clauses(formula: str) -> List[Clause]:
    try:
        tree = Parser.parse(formula)
        transformed_tree = tree.apply_transformations()  # До cnf, но apply_transformations теперь без финального cnf
        skolem_tree = transformed_tree.skolemize()
        cnf_tree = skolem_tree.convert_to_cnf()  # Добавлено: преобразование в CNF после сколемизации
        clauses = cnf_tree.to_cnf_clauses()
        return clauses
    except Exception as e:
        print(f"Ошибка при преобразовании формулы '{formula}' в клаузы: {str(e)}")
        return []


def simplify_clauses(clauses: List[Clause], max_iterations: int = 50) -> List[Clause]:
    simplified = [c for c in clauses if not c.is_tautology()]
    
    changed = True
    iteration = 0
    
    while changed and iteration < max_iterations:
        iteration += 1
        changed = False
        
        units = [c for c in simplified if len(c.literals) == 1]
        new_simplified = units.copy()  # Сохраняем units
        
        # Проверка на противоречие среди units
        for i in range(len(units)):
            unit_lit1 = next(iter(units[i].literals))
            for j in range(i+1, len(units)):
                unit_lit2 = next(iter(units[j].literals))
                if unit_lit1.is_complementary(unit_lit2):
                    mgu = unit_lit1.most_general_unifier(unit_lit2.negate())
                    if mgu is not None:
                        return [Clause(set())]
        
        for clause in [c for c in simplified if len(c.literals) > 1]:
            new_literals = clause.literals.copy()
            clause_changed = False
            
            for unit in units:
                unit_lit = next(iter(unit.literals))
                
                # Проверяем на наличие комплементарного литерала
                complementary_lits = [lit for lit in new_literals if unit_lit.is_complementary(lit)]
                for comp_lit in complementary_lits:
                    mgu = unit_lit.most_general_unifier(comp_lit)
                    if mgu is not None:
                        # Применяем MGU ко всей клаузе
                        new_literals = {l.substitute(mgu) for l in new_literals if not l.substitute(mgu).is_complementary(unit_lit.substitute(mgu))}
                        clause_changed = True
                        changed = True
                        if not new_literals:
                            return [Clause(set())]
                        break  # После одной подстановки переходим к следующему unit
                
                if clause_changed:
                    break
            
            if new_literals and len(new_literals) == 1:
                units.append(Clause(new_literals))
            
            if new_literals:
                new_simplified.append(Clause(new_literals))
        
        if changed:
            simplified = new_simplified
    
    return simplified


def standardize_variables(clause: Clause, used_names: Optional[Set[str]] = None) -> Clause:
    if used_names is None:
        used_names = set()
    
    substitution = {}
    counter = 0
    
    def fresh_var():
        nonlocal counter
        while True:
            name = f"v{counter}"
            counter += 1
            if name not in used_names:
                used_names.add(name)
                return name
    
    new_literals = set()
    for lit in clause.literals:
        new_args = []
        for arg in lit.args:
            if isinstance(arg, Variable):
                old_name = arg.name
                if old_name not in substitution:
                    substitution[old_name] = Variable(fresh_var())
                new_args.append(substitution[old_name])
            else:
                new_args.append(arg)
        new_lit = Literal(lit.predicate, new_args, lit.negated)
        new_literals.add(new_lit)
    
    return Clause(new_literals)


def resolve(kb_clauses: List[Clause], sos_clauses: List[Clause], max_clauses: int = 1000, max_steps: int = 10000) -> Tuple[bool, List[Clause], List[Tuple[Clause, Clause, Clause, Dict[str, Term]]]]:
    standardized_kb = []
    standardized_sos = []
    used_vars = set()
    
    for clause in kb_clauses:
        if not clause.is_empty():
            standardized_kb.append(standardize_variables(clause, used_vars))
    
    for clause in sos_clauses:
        if not clause.is_empty():
            standardized_sos.append(standardize_variables(clause, used_vars))
    
    kb_set = set(standardized_kb)
    sos_set = set(standardized_sos)
    all_clauses = list(kb_set.union(sos_set))
    history = []
    
    steps = 0
    
    while sos_set and steps < max_steps and len(all_clauses) < max_clauses:
        new_sos = []
        
        for sos_clause in list(sos_set):
            for kb_clause in list(kb_set.union(sos_set)):
                steps += 1
                
                c1_std = standardize_variables(sos_clause, used_vars.copy())
                c2_std = standardize_variables(kb_clause, used_vars.copy())
                
                resolvents = c1_std.resolve_with(c2_std)
                
                for new_clause_raw, mgu in resolvents:
                    if new_clause_raw.is_empty():
                        history.append((sos_clause, kb_clause, new_clause_raw, mgu))
                        return True, all_clauses + [new_clause_raw], history
                    
                    new_clause = standardize_variables(new_clause_raw, used_vars)
                    
                    if new_clause.is_tautology() or any(new_clause.literals.issubset(existing.literals) for existing in all_clauses):
                        continue
                    
                    if new_clause not in all_clauses:
                        new_sos.append(new_clause)
                        history.append((sos_clause, kb_clause, new_clause, mgu))
                        all_clauses.append(new_clause)
        
        sos_set.update(new_sos)
    
    return False, all_clauses, history


def prove_theorem(statements: List[str], goal: str) -> Tuple[bool, List[Clause], List[Tuple[Clause, Clause, Clause, Dict[str, Term]]]]:
    kb_clauses = []
    for stmt in statements:
        kb_clauses.extend(convert_to_clauses(stmt))
    
    negated_goal = f"¬({goal})"
    sos_clauses = convert_to_clauses(negated_goal)
    
    return resolve(kb_clauses, sos_clauses, max_clauses=500, max_steps=500)


def parse_statements_and_goal(data: Dict[str, Any]) -> Tuple[List[str], str]:
    if 'statements' not in data or 'goal' not in data:
        raise ValueError("JSON данные должны содержать поля 'statements' и 'goal'")
    
    statements = data['statements']
    goal = data['goal']
    
    if not isinstance(statements, list) or not all(isinstance(s, str) for s in statements):
        raise ValueError("Поле 'statements' должно быть списком строк")
    
    if not isinstance(goal, str):
        raise ValueError("Поле 'goal' должно быть строкой")
    
    return statements, goal


def resolution_proof(kb: Tuple[List[str], str], log_stream: Optional[io.StringIO] = None) -> Tuple[bool, str]:
    if log_stream is None:
        log_stream = io.StringIO()
    
    statements, goal = kb
    
    log_stream.write("ЛОГИЧЕСКОЕ ДОКАЗАТЕЛЬСТВО\n")
    
    log_stream.write("ИСХОДНЫЕ ДАННЫЕ\n")
    log_stream.write(f"Утверждений: {len(statements)}\n")
    for i, stmt in enumerate(statements, 1):
        log_stream.write(f"  {i}. {stmt}\n")
    log_stream.write(f"Цель: {goal}\n\n")
    
    log_stream.write("ПРЕОБРАЗОВАНИЯ\n")
    
    kb_clauses = []
    for i, stmt in enumerate(statements, 1):
        log_stream.write(f"\nУтверждение {i}: {stmt}\n")
        try:
            pnf = convert_formula_to_pnf(stmt)
            log_stream.write(f"ПНФ: {pnf}\n")
            
            skolem = convert_formula_to_skolem(stmt)
            log_stream.write(f"Сколемизация: {skolem}\n")
            
            clauses = convert_to_clauses(stmt)
            log_stream.write(f"Клаузы ({len(clauses)}):\n")
            for j, clause in enumerate(clauses, 1):
                log_stream.write(f"{j}. {clause}\n")
            
            kb_clauses.extend(clauses)
        except Exception as e:
            log_stream.write(f"ОШИБКА: {str(e)}\n")
    
    log_stream.write(f"\nОтрицание цели: ¬({goal})\n")
    negated_clauses = []
    try:
        negated_goal = f"¬({goal})"
        pnf_goal = convert_formula_to_pnf(negated_goal)
        log_stream.write(f"ПНФ отрицания цели: {pnf_goal}\n")
        
        skolem_goal = convert_formula_to_skolem(negated_goal)
        log_stream.write(f"Сколемизация отрицания цели: {skolem_goal}\n")
        
        negated_clauses = convert_to_clauses(negated_goal)
        log_stream.write(f"Клаузы из отрицания цели ({len(negated_clauses)}):\n")
        for j, clause in enumerate(negated_clauses, 1):
            log_stream.write(f"      {j}. {clause}\n")
        
    except Exception as e:
        log_stream.write(f"ОШИБКА: {str(e)}\n")
    
    all_clauses = kb_clauses + negated_clauses
    
    log_stream.write(f"\nСТАТИСТИКА ДО УПРОЩЕНИЯ\n")
    log_stream.write(f"Всего клауз: {len(all_clauses)}\n")
    for i, clause in enumerate(all_clauses, 1):
        log_stream.write(f"  {i}. {clause}\n")
    
    log_stream.write(f"\nУПРОЩЕНИЕ КЛАУЗ\n")
    try:
        simplified_kb = simplify_clauses(kb_clauses)
        simplified_sos = simplify_clauses(negated_clauses)
        simplified_clauses = simplified_kb + simplified_sos
        log_stream.write(f"Клауз после упрощения: {len(simplified_clauses)}\n")
        for i, clause in enumerate(simplified_clauses, 1):
            log_stream.write(f"{i}. {clause}\n")
    except Exception as e:
        log_stream.write(f"ОШИБКА при упрощении: {str(e)}\n")
        simplified_kb = kb_clauses
        simplified_sos = negated_clauses
    
    log_stream.write(f"\nМЕТОД РЕЗОЛЮЦИЙ\n")
    try:
        success, final_clauses, history = resolve(simplified_kb, simplified_sos, max_clauses=500, max_steps=500)
        
        log_stream.write(f"\nРЕЗУЛЬТАТ: {'ДОКАЗАНО' if success else 'НЕ ДОКАЗАНО'}\n")
        log_stream.write(f"Статистика:\n")
        log_stream.write(f"Всего клауз: {len(final_clauses)}\n")
        log_stream.write(f"Шагов резолюции: {len(history)}\n")
        
        if history:
            log_stream.write(f"\nДЕТАЛЬНАЯ ИСТОРИЯ РЕЗОЛЮЦИЙ\n")
            for i, (parent1, parent2, resolvent, substitution) in enumerate(history, 1):
                log_stream.write(f"\nШАГ {i}\n")
                log_stream.write(f"Родитель 1: {parent1}\n")
                log_stream.write(f"Родитель 2: {parent2}\n")
                log_stream.write(f"Результат: {resolvent}\n")
                
                if substitution:
                    subst_items = [f"{var} = {term.get_name()}" for var, term in substitution.items()]
                    subst_str = ", ".join(subst_items)
                    log_stream.write(f"Подстановка: {subst_str}\n")
                else:
                    log_stream.write(f"Подстановка: (пустая)\n")
                
                if resolvent.is_empty():
                    log_stream.write(f"ОБНАРУЖЕНО ПРОТИВОРЕЧИЕ!\n")
                    break
        
        if not success:
            log_stream.write(f"\nКОНЕЧНЫЕ КЛАУЗЫ\n")
            for i, clause in enumerate(final_clauses[:15], 1):
                log_stream.write(f"  {i}. {clause}\n")
            if len(final_clauses) > 15:
                log_stream.write(f"  ... и еще {len(final_clauses) - 15} клауз\n")
    
    except Exception as e:
        log_stream.write(f"\nКРИТИЧЕСКАЯ ОШИБКА в методе резолюций:\n{str(e)}\n")
        success = False
    
    log_stream.write("КОНЕЦ ДОКАЗАТЕЛЬСТВА\n")
    
    return success, log_stream.getvalue()


def solve_logic_task(data: dict) -> str:
    try:
        statements, goal = parse_statements_and_goal(data)
        
        success, log = resolution_proof((statements, goal))
        
        result = {
            "success": 1 if success else 0,
            "log": log
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2, separators=(',', ': '))
    
    except Exception as e:
        error_log = io.StringIO()
        error_log.write("ОШИБКА ВЫПОЛНЕНИЯ\n")
        error_log.write(f"Тип ошибки: {type(e).__name__}\n")
        error_log.write(f"Сообщение: {str(e)}\n\n")
        error_log.write("Контекст:\n")
        error_log.write(f"Данные: {json.dumps(data, ensure_ascii=False, indent=2)}\n")
        
        import traceback
        error_log.write(traceback.format_exc())
        
        error_result = {
            "success": 0,
            "log": error_log.getvalue()
        }
        
        return json.dumps(error_result, ensure_ascii=False, indent=2, separators=(',', ': '))


def main():
    data = {'statements': ['∀(x, ∀(y, depends_on(x, y) → ¬depends_on(y, x)))', 'depends_on(Petya, Vasya)'], 'goal': '¬depends_on(Vasya, Petya)'}
    
    result = solve_logic_task(data)
    result = json.loads(result)
    print(result['log'])


if __name__ == "__main__":
    main()