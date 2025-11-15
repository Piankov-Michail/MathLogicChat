import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.resolution_method import parse_statements_and_goal, resolution_proof

def test_socrates_mortality():
    """Тест: Смертность Сократа - классический силлогизм"""
    data = {
        "statements": [
            "∀(x, Human(x) → Mortal(x))",
            "Human(Socrates)"
        ],
        "goal": "Mortal(Socrates)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что Сократ смертен"

def test_no_doctor_is_healer():
    """Тест: Ни один врач не является целителем"""
    data = {
        "statements": [
            "∃(p, Patient(p) ∧ ∀(d, Doctor(d) → Loves(p, d)))",
            "∀(p, Patient(p) → ∀(h, Healer(h) → ¬Loves(p, h)))"
        ],
        "goal": "∀(x, Doctor(x) → ¬Healer(x))"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что врач не может быть целителем"

def test_everyone_has_friend():
    """Тест: У каждого есть друг"""
    data = {
        "statements": [
            "∀(x, Person(x) → ∃(y, Loves(y, x)))",
            "∀(x, y, Loves(x, y) → Friend(x, y))"
        ],
        "goal": "∀(x, Person(x) → ∃(y, Friend(y, x)))"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что у каждого человека есть друг"

def test_transitive_relationship():
    """Тест: Транзитивность отношения 'больше'"""
    data = {
        "statements": [
            "∀(x, y, z, (Greater(x, y) ∧ Greater(y, z)) → Greater(x, z))",
            "Greater(Five, Three)",
            "Greater(Three, One)"
        ],
        "goal": "Greater(Five, One)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать транзитивность: 5 > 1"

def test_no_circular_dependency():
    """Тест: Отсутствие циклических зависимостей"""
    data = {'statements': ['∀(x, ∀(y, depends_on(x, y) → ¬depends_on(y, x)))', 'depends_on(Petya, Vasya)'], 'goal': '¬depends_on(Vasya, Petya)'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать отсутствие обратной зависимости"

def test_inheritance_hierarchy():
    """Тест: Иерархия наследования"""
    data = {
        "statements": [
            "∀(x, Cat(x) → Animal(x))",
            "∀(x, Animal(x) → LivingBeing(x))",
            "Cat(Murzik)"
        ],
        "goal": "LivingBeing(Murzik)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать наследование свойств"

def test_contradiction_detection():
    """Тест: Обнаружение противоречия должно завершиться неудачей"""
    data = {
        "statements": [
            "∀(x, Human(x) → Mortal(x))",
            "Human(Socrates)"
        ],
        "goal": "¬Mortal(Socrates)"  # Ложная цель
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is False, "Доказательство должно провалиться для ложной цели"