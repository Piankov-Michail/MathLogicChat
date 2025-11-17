import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.resolution_method import parse_statements_and_goal, resolution_proof

def test_socrates_mortality():
    """Тест: Смертность Сократа - классический силлогизм"""
    # 1) Все люди смертны
    # 2) Сократ - человек
    # То что надо доказать: Сократ смертен
    data = {
        "statements": [
            "all(x, human(x) → mortal(x))",
            "human(socrates)"
        ],
        "goal": "mortal(socrates)"
    }

    kb = parse_statements_and_goal(data)
    success = resolution_proof(kb)

    assert success is True, "Должны доказать, что Сократ смертен"

def test_no_doctor_is_healer():
    """Тест: Ни один врач не является целителем"""
    # 1) Некоторые пациенты любят всех врачей
    # 2) Все пациенты не любят целителей
    # То что надо доказать: Врач не может быть целителем
    data = {
        "statements": [
            "some(p, patient(p) ∧ all(d, doctor(d) → loves(p, d)))",
            "all(p, z, (patient(p) ∧ healer(z)) → ¬loves(p, z))"
        ],
        "goal": "all(d, z, (doctor(d) ∧ healer(z)) → d ≠ z)"
    }

    kb = parse_statements_and_goal(data)
    success = resolution_proof(kb)

    assert success is True, "Должны доказать, что врач ≠ целитель"

def test_everyone_has_friend():
    """Тест: У каждого есть друг"""
    # 1) Для каждого человека существует кто-то, кто его любит
    # 2) Если кто-то тебя любит, то он твой друг
    # То что надо доказать: У каждого есть друг
    data = {
        "statements": [
            "all(x, person(x) → some(y, loves(y, x)))",
            "all(x, y, loves(x, y) → friend(x, y))"
        ],
        "goal": "all(x, person(x) → some(y, friend(y, x)))"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что у каждого человека есть друг"

def test_transitive_relationship():
    """Тест: Транзитивность отношения 'больше'"""
    # 1) Если A > B и B > C, то A > C
    # 2) 5 > 3
    # 3) 3 > 1
    # То что надо доказать: 5 > 1
    data = {
        "statements": [
            "all(x, y, z, (greater(x, y) ∧ greater(y, z)) → greater(x, z))",
            "greater(five, three)",
            "greater(three, one)"
        ],
        "goal": "greater(five, one)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать транзитивность: 5 > 1"

def test_no_circular_dependency():
    """Тест: Отсутствие циклических зависимостей"""
    # 1) Если A зависит от B, то B не зависит от A
    # 2) X зависит от Y
    # То что надо доказать: Y не зависит от X
    data = {
        "statements": [
            "all(x, y, depends_on(x, y) → ¬depends_on(y, x))",
            "depends_on(X, Y)"
        ],
        "goal": "¬depends_on(Y, X)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать отсутствие обратной зависимости"

def test_inheritance_hierarchy():
    """Тест: Иерархия наследования"""
    # 1) Все кошки - животные
    # 2) Все животные - живые существа
    # 3) Мурзик - кошка
    # То что надо доказать: Мурзик - живое существо
    data = {
        "statements": [
            "all(x, cat(x) → animal(x))",
            "all(x, animal(x) → living_being(x))",
            "cat(murzik)"
        ],
        "goal": "living_being(murzik)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать наследование свойств"

def test_contradiction_detection():
    """Тест: Обнаружение противоречия должно завершиться неудачей"""
    # 1) Все люди смертны
    # 2) Сократ - человек
    # То что надо доказать: Сократ бессмертен (ложное утверждение)
    data = {
        "statements": [
            "all(x, human(x) → mortal(x))",
            "human(socrates)"
        ],
        "goal": "¬mortal(socrates)"  # Ложная цель
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is False, "Доказательство должно провалиться для ложной цели"