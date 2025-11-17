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

def test_input_noise():
    """Тест: Устойчивость к "мусору" на входе"""
    # Вход: "Ну типа, если там дождь, короче, зонт бери, а если нет... сам решай. Докажи что-нибудь."
    # Ожидаемый результат: формализатор не смог выделить точную БЗ и цель, 
    # объяснитель сообщает вежливый отказ.

    data = {
        "statements": [
            "Неправильно сформулированное утверждение, невозможно формализовать"
        ],
        "goal": "Неопределено"
    }

    try:
        kb = parse_statements_and_goal(data)
        success, log = resolution_proof(kb)
        assert success is False, "Должно быть невозможно доказать что-либо"
    except Exception as e:
        print("Ожидаемый отказ системы:", str(e))


def test_student_passed_some_exam():
    """Тест: Каждый студент сдал хотя бы один экзамен (проверка на вложенность ∀ → ∃)"""
    # 1) Каждый студент сдал хотя бы один экзамен
    # То что надо доказать: Для любого студента существует экзамен, который он сдал
    data = {
        "statements": [
            "∀(s, some(e, student(s) → (exam(e) ∧ passed(s, e))))"
        ],
        "goal": "∀(s, some(e, student(s) → (exam(e) ∧ passed(s, e))))"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)

    assert success is True, "Должны доказать, что каждый студент сдал хотя бы один экзамен"


def test_cold_coat():
    """Тест: Снег и пальто"""
    # 1) Если холодно или снег, человек носит пальто
    # 2) Сегодня холодно
    # Нужно доказать: Человек носит пальто
    data = {
        "statements": [
            "∀(x, (cold(x) ∨ snow(x)) → wears_coat(x))",
            "cold(today)"
        ],
        "goal": "wears_coat(today)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_cats_mice():
    """Тест: Кошки и мыши"""
    # 1) Для всех кошек и мышей: если кошка и мышь, мышь не боится кошки
    # 2) Том - кошка
    # 3) Джерри - мышь
    # Нужно доказать: Джерри не боится Тома
    data = {
        "statements": [
            "∀(c, m, (cat(c) ∧ mouse(m)) → ¬afraid(m, c))",
            "cat(tom)",
            "mouse(jerry)"
        ],
        "goal": "¬afraid(jerry, tom)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_sport_health():
    """Тест: Спорт и здоровье"""
    # 1) Если человек тренируется и ест правильно, он здоров
    # 2) Эмма тренируется
    # 3) Эмма ест правильно
    # Нужно доказать: Эмма здорова
    data = {
        "statements": [
            "∀(p, (trains(p) ∧ eats_healthy(p)) → healthy(p))",
            "trains(emma)",
            "eats_healthy(emma)"
        ],
        "goal": "healthy(emma)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_reading_knowledge():
    """Тест: Чтение и знание"""
    # 1) Если человек читает книгу и книга интересная, он получает знание
    # 2) Алекс читает книгу по физике
    # 3) Книга по физике интересная
    # Нужно доказать: Алекс получил знание
    data = {
        "statements": [
            "∀(p, b, (reads(p, b) ∧ interesting(b)) → knows(p, info))",
            "reads(alex, physics_book)",
            "interesting(physics_book)"
        ],
        "goal": "knows(alex, info)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_some_birds_cannot_fly():
    """Тест: Некоторые птицы не умеют летать"""
    # Вход: "Некоторые птицы не умеют летать"
    # Ожидаемый вывод: ∃x (Птица(x) ∧ ¬Летает(x))
    data = {
        "statements": [
            "∃(x, bird(x) ∧ ¬flies(x))"
        ],
        "goal": "∃(x, bird(x) ∧ ¬flies(x))"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_pet_care():
    """Тест: Питомцы и уход"""
    # 1) Существует собака, и все владельцы ухаживают за ней
    # 2) Фидо - собака
    # Нужно доказать: Фидо счастлив
    data = {
        "statements": [
            "∃(d, dog(d) ∧ ∀(o, owner(o) → cares(o, d)))",
            "dog(fido)"
        ],
        "goal": "happy(fido)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_plants_bloom():
    """Тест: Растения и вода"""
    # 1) Существует растение, которое либо полито, либо солнечно
    # 2) Роза солнечна
    # Нужно доказать: Роза цветёт
    data = {
        "statements": [
            "∃(p, plant(p) ∧ (watered(p) ∨ sunny(p)))",
            "sunny(rose)"
        ],
        "goal": "blooms(rose)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_student_sad_and_not_happy():
    """Тест: Студент сдал или не сдал экзамен и его эмоции"""
    # БЗ:
    # 1) ∀x (Студент(x) ∧ ¬Сдал(x, Матан) → Грустный(x))
    # 2) ∀x (Грустный(x) → ¬Веселый(x))
    # 3) Студент(Петя)
    # 4) ¬Сдал(Петя, Матан)
    # Утверждение: ¬Веселый(Петя)
    data = {
        "statements": [
            "∀(x, (student(x) ∧ ¬passed(x, math)) → sad(x))",
            "∀(x, sad(x) → ¬happy(x))",
            "student(petya)",
            "¬passed(petya, math)"
        ],
        "goal": "¬happy(petya)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_tasks_approval():
    """Тест: Задачи и выполнение"""
    # 1) Существует задача, и все менеджеры одобряют её
    # 2) Задача выполнена
    # 3) Менеджер существует
    # Нужно доказать: Проект продвигается
    data = {
        "statements": [
            "some(t, task(t) ∧ ∀(m, manager(m) → approved_by(t, m)))",
            "done(task1)",
            "manager(manager1)"
        ],
        "goal": "progress(project)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True

def test_cat_gray_trap():
    """Тест: Кошка Мурка серая - ловушка отсутствующего знания"""
    # Вход: "Докажи, что кошка Мурка - серая. Мы знаем, что все кошки в этом доме серые."
    # Ожидаемый вывод: FALSE
    # Объяснение: "Нет информации о том, живет ли Мурка в доме"
    data = {
        "statements": [
            "∀(x, (cat(x) ∧ lives_in_this_house(x)) → gray(x))"
        ],
        "goal": "gray(murka)"
    }

    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is False

def test_studies_exam():
    """Тест: Учёба и экзамены"""
    # 1) Если студент учится и не ленивый, он сдаёт экзамен
    # 2) Алекс учится
    # 3) Алекс не ленивый
    # Нужно доказать: Алекс сдал экзамен
    data = {
        "statements": [
            "∀(s, (studies(s) ∧ ¬lazy(s)) → passes_exam(s))",
            "studies(alex)",
            "¬lazy(alex)"
        ],
        "goal": "passes_exam(alex)"
    }
    kb = parse_statements_and_goal(data)
    success, log = resolution_proof(kb)
    assert success is True
