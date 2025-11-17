import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.resolution_method import parse_statements_and_goal, resolution_proof

def test_socrates_mortality():
    """Тест: Смертность Сократа - классический силлогизм"""
    # 1) Все люди смертны
    # 2) Сократ - человек
    # То что надо доказать: Сократ смертен
    data = {'statements': ['∀(x, human(x) → mortal(x))', 'human(Socrates)'], 'goal': 'mortal(Socrates)'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что Сократ смертен"

def test_contradiction_detection():
    """Тест: Обнаружение противоречия должно завершиться неудачей"""
    # 1) Все люди смертны
    # 2) Сократ - человек
    # То что надо доказать: Сократ бессмертен (ложное утверждение)
    data = {'statements': ['∀(x, human(x) → mortal(x))', 'human(Socrates)'], 'goal': '¬mortal(Socrates)'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is False, "Доказательство должно провалиться для ложной цели"

def test_modus_tollens():
    """Тест: Modus Tollens должно доказать утверждение"""
    # 1) Если не идет дождь, то Саша идет гулять
    # 2) Саша не идет гулять
    # Доказать: Идет дождь
    data = {'statements': ['∀(x, ¬rain(x) → walk(Sasha, x))', '¬∃(x, walk(Sasha, x))'], 'goal': '∃(x, rain(x))'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is True, "Должно доказать дождь через Modus Tollens"

def test_no_doctor_is_healer():
    """Тест: Ни один врач не является целителем"""
    # 1) Некоторые пациенты любят всех врачей
    # 2) Все пациенты не любят целителей
    # То что надо доказать: Врач не может быть целителем
    data = {'statements': ['∃(p, patient(p) ∧ ∀(d, doctor(d) → loves(p, d)))', '∀(p, patient(p) → ∀(h, healer(h) → ¬loves(p, h)))'], 'goal': '∀(x, doctor(x) → ¬healer(x))'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что врач ≠ целитель"

def test_vaccination():
    """Тест: Доказательство отсутствия заболеваний"""
    # Привитые люди не болеют
    # Непривитые люди не болеют
    # Доказать: Никто не болеет
    data = {'statements': ['∀(x, vaccinated(x) → ¬sick(x))', '∀(x, ¬vaccinated(x) → ¬sick(x))'], 'goal': '∀(x, ¬sick(x))'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is True, "Должно доказать отсутствие заболеваний у всех"

def test_everyone_has_friend():
    """Тест: У каждого есть друг"""
    # 1) Для каждого человека существует кто-то, кто его любит
    # 2) Если кто-то тебя любит, то он твой друг
    # То что надо доказать: У каждого есть друг
    data = {'statements': ['∀(x, person(x) → ∃(y, loves(y, x)))', '∀(x, ∀(y, loves(y, x) → friend(y, x)))'], 'goal': '∀(x, person(x) → ∃(y, friend(y, x)))'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что у каждого человека есть друг"

def test_everyone_has_child_relation():
    """Тест: Доказательство отношения 'быть ребенком'"""
    # У каждого человека есть родитель
    # Если y является родителем x, то x является ребенком y
    # Доказать: У каждого человека есть тот, для кого он является ребенком
    data = {'statements': ['∀(x, person(x) → ∃(y, parent(y, x)))', '∀(x, ∀(y, parent(y, x) → child(x, y)))'], 'goal': '∀(x, person(x) → ∃(y, child(x, y)))'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is True, "Должно доказать существование обратного отношения"

def test_inheritance_hierarchy():
    """Тест: Иерархия наследования"""
    # 1) Все кошки - животные
    # 2) Все животные - живые существа
    # 3) Мурзик - кошка
    # То что надо доказать: Мурзик - живое существо
    data = {'statements': ['∀(x, cat(x) → animal(x))', '∀(x, animal(x) → living_being(x))', 'cat(Murzik)'], 'goal': 'living_being(Murzik)'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать наследование свойств"

def test_contradiction_implies_anything():
    """Тест: Из противоречия следует любое утверждение"""
    # Кошка - это млекопитающее
    # Кошка - не млекопитающее
    # Доказать: Луна сделана из сыра
    data = {'statements': ['∀(x, cat(x) → mammal(x))', '∀(x, cat(x) → ¬mammal(x))'], 'goal': '∀(x, moon(x) → made_of_cheese(x))'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is False, "Из противоречия должно следовать любое утверждение"

def test_false_generalization():
    """Тест: Ложное обобщение должно завершиться неудачей"""
    # Все тигры являются полосатыми
    # Все кошки являются полосатыми
    # Доказать: Все кошки являются тиграми
    data = {'statements': ['∀(x, tiger(x) → striped(x))', '∀(x, cat(x) → striped(x))'], 'goal': '∀(x, cat(x) → tiger(x))'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is False, "Не должно доказывать ложное включение"

def test_penguin_flying():
    """Тест: Частный случай из отрицания общего утверждения"""
    # 1) Неверно, что все птицы умеют летать
    # 2) Пингвин - это птица
    # Доказать: Пингвин не умеет летать
    data = {'statements': ['¬∀x(bird(x) → fly(x))', 'bird(penguin)'], 'goal': '¬fly(penguin)'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is False, "Не должно доказывать частный случай без дополнительных предпосылок"

def test_transitive_relationship():
    """Тест: Транзитивность отношения 'больше'"""
    # 1) Если A > B и B > C, то A > C
    # 2) 5 > 3
    # 3) 3 > 1
    # То что надо доказать: 5 > 1
    data = {'statements': ['∀(a, ∀(b, ∀(c, (greater(a, b) ∧ greater(b, c)) → greater(a, c))))', 'greater(5, 3)', 'greater(3, 1)'], 'goal': 'greater(5, 1)'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать транзитивность: 5 > 1"

def test_no_circular_dependency():
    """Тест: Отсутствие циклических зависимостей"""
    # 1) Если A зависит от B, то B не зависит от A
    # 2) X зависит от Y
    # То что надо доказать: Y не зависит от X
    data = {'statements': ['∀(a, ∀(b, depends(a, b) → ¬depends(b, a)))', 'depends(X, Y)'], 'goal': '¬depends(Y, X)'}

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать отсутствие обратной зависимости"

def test_maximal_number():
    """Тест: Доказательство отсутствия максимального числа"""
    # Для каждого числа существует большее число
    # Если число A больше числа B, то B не больше A
    # Доказать: Не существует числа, которое является наибольшим
    data = {'statements': ['∀(x, number(x) → ∃(y, number(y) ∧ greater(y, x)))', '∀(x, ∀(y, (number(x) ∧ number(y) ∧ greater(x, y)) → ¬greater(y, x)))'], 'goal': '¬∃(x, number(x) ∧ ∀(y, number(y) → ¬greater(y, x)))'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is True, "Должно доказать отсутствие максимального числа"

def test_syllogism():
    """Тест: Простой силлогизм"""
    # Все A являются B
    # Все B являются C
    # Доказать: Все A являются C
    data = {'statements': ['∀(x, A(x) → B(x))', '∀(x, B(x) → C(x))'], 'goal': '∀(x, A(x) → C(x))'}
    
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    
    assert success is True, "Должно доказать транзитивность включения"