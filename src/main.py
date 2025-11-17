import os
import sys
from pathlib import Path
os.environ["KIVY_NO_ARGS"] = "1"
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch')

from openai import OpenAI
from typing import Generator
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock, mainthread
from kivy.metrics import dp, sp
from kivy.core.window import Window
from kivy.core.clipboard import Clipboard
from kivy.core.text import LabelBase
from kivy.graphics import Color, Rectangle
from threading import Thread
import time
import uuid
import json

import re
from src.resolution_method import solve_logic_task

KEY_M = 1084   # м → V
KEY_F = 1092   # ф → A
KEY_YA = 1103  # я → Z
KEY_S = 1089   # с → C

if getattr(sys, 'frozen', False):
    exe_dir = Path(sys.executable).parent
    resources_dir = Path(sys._MEIPASS)
else:
    exe_dir = Path(__file__).parent.parent
    resources_dir = exe_dir

font_dir = resources_dir / "fonts"
data_dir = exe_dir / "data"

chats_dir = data_dir / "chats"
config_file = data_dir / "config.json"

os.makedirs(chats_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

try:
    LabelBase.register(
        name='Emoji',
        fn_regular=os.path.join(font_dir, 'NotoColorEmoji.ttf')
    )
except Exception as e:
    print("Не удалось загрузить Emoji шрифт:", e)

try:
    LabelBase.register(
        name='Roboto',
        fn_regular=os.path.join(font_dir, 'DejaVuSans.ttf')
    )
except Exception as e:
    print("Не удалось загрузить Roboto шрифт:", e)


class LLM:
    def __init__(
        self,
        token: str,
        model: str = "",
        base_url: str = "",
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 65536):
        self.token = token
        self.model = model
        self.base_url = base_url.strip()
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=self.base_url, api_key=self.token)
        self.streaming_chat_id = None

    def make_query_generator(self, user_query: str, system_prompt: str = "/think", tools=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=True,
            tools=tools,
            tool_choice="auto"
        )

        full_content = ""
        tool_calls = []

        for chunk in completion:
            delta = chunk.choices[0].delta

            if delta.content:
                full_content += delta.content
                yield {"type": "text", "content": delta.content}

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    index = tc_delta.index
                    if len(tool_calls) <= index:
                        tool_calls.append({"id": None, "function": {"name": "", "arguments": ""}})
                    tc = tool_calls[index]
                    if tc["id"] is None:
                        tc["id"] = tc_delta.id
                    if tc_delta.function.name:
                        tc["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc["function"]["arguments"] += tc_delta.function.arguments

        if tool_calls:
            for tc in tool_calls:
                yield {"type": "tool_call", "tool_call": tc}


class NoDragScrollView(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._touch_initial_pos = None
        self._touch_initial_scroll_y = None
        self._is_scrolling_bar = False

    def on_touch_down(self, touch):
        if self.vbar[0] < 1:
            vbar_x_start = self.x + self.width - self.bar_width - self.bar_margin
            vbar_x_end = vbar_x_start + self.bar_width
            vbar_height = self.vbar[1] * self.height
            vbar_y_start = self.y + self.vbar[0] * (self.height - vbar_height)
            vbar_y_end = vbar_y_start + vbar_height
            
            if (vbar_x_start <= touch.x <= vbar_x_end and 
                vbar_y_start <= touch.y <= vbar_y_end and
                touch.button == 'left'):
                self._is_scrolling_bar = True
                self._touch_initial_pos = touch.y
                self._touch_initial_scroll_y = self.scroll_y
                return True
        
        if hasattr(touch, 'button') and touch.button in ('left', 'right'):
            original_do_scroll_y = self.do_scroll_y
            original_do_scroll_x = self.do_scroll_x
            self.do_scroll_y = False
            self.do_scroll_x = False

            result = super().on_touch_down(touch)
            self.do_scroll_y = original_do_scroll_y
            self.do_scroll_x = original_do_scroll_x

            return result

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._is_scrolling_bar and touch.button == 'left':
            content_height = 0
            if self.children:
                content_height = self.children[0].height
            
            viewport_height = self.height
            max_scroll = max(0, content_height - viewport_height)
            
            if max_scroll > 0:
                delta_y = touch.y - self._touch_initial_pos
                
                track_height = viewport_height - (self.vbar[1] * viewport_height)
                
                if track_height > 0:
                    scroll_ratio = delta_y / track_height
                    new_scroll_y = self._touch_initial_scroll_y + scroll_ratio
                    self.scroll_y = max(0, min(1, new_scroll_y))
            return True
        
        return super().on_touch_move(touch)
    def on_touch_up(self, touch):
        if self._is_scrolling_bar and touch.button == 'left':
            self._is_scrolling_bar = False
            self._touch_initial_pos = None
            self._touch_initial_scroll_y = None
            return True
        
        return super().on_touch_up(touch)

    def on_scroll_wheel(self, touch, value):
        return super().on_scroll_wheel(touch, value * 3)


class SelectableLabel(TextInput):
    def __init__(self, message: str = "", **kwargs):
        self.message = message
        kwargs.setdefault('size_hint_y', None)
        kwargs.setdefault('size_hint_x', 1)
        kwargs.setdefault('readonly', True)
        kwargs.setdefault('multiline', True)
        kwargs.setdefault('text', message)
        kwargs.setdefault('font_size', sp(16))
        kwargs.setdefault('padding', [dp(12), dp(8)])
        kwargs.setdefault('background_color', (0, 0, 0, 0))
        kwargs.setdefault('foreground_color', (1, 1, 1, 1))
        kwargs.setdefault('halign', 'left')
        kwargs.setdefault('use_bubble', True)
        kwargs.setdefault('use_handles', True)
        kwargs.setdefault('cursor_blink', False)
        kwargs.setdefault('cursor_width', 0)
        kwargs.setdefault('do_wrap', True)

        super().__init__(**kwargs)
        self.bind(minimum_height=self._update_height)
        self.bind(on_touch_down=self.on_label_touch_down)
        self._update_height()

    def _update_height(self, *args):
        self.height = max(dp(40), self.minimum_height)

    def update_width(self, new_width):
        self.width = new_width
    
    def on_label_touch_down(self, instance, touch):
        if self.collide_point(*touch.pos):
            self.focus = True

    def on_focus(self, instance, value):
        app = App.get_running_app()
        if value:
            app.focused_message = self
        else:
            if app.focused_message is self:
                app.focused_message = None


class FixedTextInput(TextInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def paste(self):
        try:
            data = Clipboard.paste()
            if data:
                self.insert_text(data)
        except Exception as e:
            print(f"Ошибка вставки: {e}")


class ChatApp(App):
    use_kivy_settings = False
    is_streaming = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_chat_id = None
        self.load_config()
        self.llm = LLM(
            token=self.config.get("token", ""),
            model=self.config.get("model", ""),
            base_url=self.config.get("base_url", "")
        )
        self.response_buffer = ""
        self.last_update_time = 0
        self.update_interval = 0.05
        self.focused_message = None
        self.ctrl_pressed = False
        self.streaming_placeholder = None

    def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "solve_logic_task":
            task = arguments.get("task_text", "").strip()
            if not task:
                return "[Ошибка: пустая задача]"
            
            try:
                formalized = self.formalize_logic_task(task)
                return formalized
            except Exception as e:
                return f"[Ошибка формализации: {str(e)}]"
        else:
            return f"[Неизвестный инструмент: {tool_name}]"

    @staticmethod
    def extract_json_from_response(response_text: str) -> str:
        match = re.search(r"```(?:json)?\s*({.*})\s*```", response_text, re.DOTALL)
        if match:
            return match.group(1)
        return response_text.strip()

    def formalize_logic_task(self, task: str) -> str:
        system_prompt = (
            "Ты — система формальной логики. Преобразуй задачу в JSON со строго определённым синтаксисом.\n"
            "Правила преобразования:\n"
            "- Используй только конструкции:\n"
            "    • Экзистенциальные: ∃(x, P(x) ∧ Q(x))\n"
            "    • Универсальные: ∀(x, P(x) → Q(x))\n"
            "    • Вложенные кванторы разрешены в теле импликации/конъюнкции, например:\n"
            "        ∃(x, P(x) ∧ ∀(y, R(y) → S(x, y)))\n"
            "        ∀(x, (P(x) ∧ ∃(y, R(x, y))) → Q(x))\n"
            "- Предикаты: только унарные (Person(x)) или бинарные (Friend(x, y)).\n"
            "  НЕ своди бинарные отношения к унарным (например, не используй LovedByDoctor(x)).\n"
            "- Операции: ∃, ∀, ∧, ∨, ¬, →, =, ≠. ЗАПРЕЩЕНЫ: ↔, ⇔, ≡, ⊕ и т.п.\n\n"
            
            "⚠️ Ключевое правило интерпретации:\n"
            "Фразы вроде «некоторые A любят B» НЕОДНОЗНАЧНЫ в русском. Выбирай интерпретацию ТАК,\n"
            "чтобы логически следовало утверждение из поля 'goal', если оно универсальное (начинается с ∀).\n"
            "Обычно это означает:\n"
            "    • «Некоторые пациенты любят докторов» → ∃(p, patient(p) ∧ ∀(d, doctor(d) → loves(p, d)))\n"
            "      (существует пациент, который любит ВСЕХ докторов),\n"
            "    а НЕ ∃p ∃d (patient(p) ∧ doctor(d) ∧ loves(p,d)),\n"
            "      потому что последнее слишком слабо для доказательства ∀-утверждений.\n"
            "    • Аналогично: «некоторые ученики уважают учителей» → ∃(s, student(s) ∧ ∀(t, teacher(t) → respects(s,t)))\n\n"
            
            "Другие типы фраз:\n"
            "    • «Каждый A любит какого-нибудь B» → ∀(x, A(x) → ∃(y, B(y) ∧ loves(x, y)))\n"
            "    • «Любой A, который любит всех B, …» → ∀(x, (A(x) ∧ ∀(y, B(y) → loves(x, y))) → …)\n"
            "    • «Ни один A не любит B» → ∀(x, A(x) → ∀(y, B(y) → ¬loves(x, y)))\n"
            "    • «Есть A, которого никто не любит» → ∃(x, A(x) ∧ ∀(y, ¬loves(y, x)))\n\n"
            
            "Примеры корректных преобразований:\n"
            "'Иван — друг Петра' → Friend(Ivan, Petr)\n"
            "'Все люди смертны' → ∀(x, human(x) → mortal(x))\n"
            "'Некоторые пациенты любят всех врачей' → ∃(p, patient(p) ∧ ∀(d, doctor(d) → loves(p, d)))\n"
            "'Некоторые пациенты любят врачей' (в задачах на доказательство) → ТО ЖЕ, что выше\n"
            "'Ни один пациент не любит знахарей' → ∀(p, patient(p) → ∀(h, healer(h) → ¬loves(p, h)))\n\n"
            
            "Формат вывода:\n"
            "- ТОЛЬКО валидный JSON (без комментариев, без пояснений):\n"
            "  {\"statements\": [\"формула1\", \"формула2\", ...], \"goal\": \"формула\"}\n"
            "- Имена предикатов — чувствительны к регистру: используй строчные буквы (loves, doctor),\n"
            "  если в примерах не указано иное. Константы (Ivan) — с большой буквы.\n"
            "- Не добавляй кванторы без необходимости; не используй скобки лишние.\n\n"
            
            "Если в задаче не хватает информации для однозначной формализации — выбирай наиболее сильную\n"
            "интерпретацию, допускающую логический вывод цели (особенно если goal — ∀-формула)."
        )

        user_prompt = f"Формализуй эту задачу:\n{task}"

        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
            stream=False
        )
        
        raw_response = response.choices[0].message.content.strip()

        clean_json_str = self.extract_json_from_response(raw_response)
        
        try:
            data = json.loads(clean_json_str)
            print(data)
            full_report = solve_logic_task(data)
            
            explanation_prompt = (
                "Ты — эксперт по логике и математике. Объясни пользователю результат решения логической задачи.\n\n"
                "Исходная задача:\n"
                f"{task}\n\n"
                "Формализованная задача:\n"
                f"{json.dumps(data, ensure_ascii=False, indent=2)}\n\n"
                "результат решения (лог алгоритма резолюции):\n"
                f"{full_report}\n\n"
                "Объясни:\n"
                "1. Что означают формальные записи в JSON (расшифруй предикаты и кванторы)\n"
                "2. Как работал метод резолюции - основные шаги доказательства\n"
                "3. Был ли найден ответ и что он означает\n"
                "4. Простой вывод на естественном языке\n\n"
                "Будь понятным и дружелюбным, используй примеры если нужно."
            )

            user_prompt = f"Объясни эту задачу:\n{task}"
            explanation_response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": explanation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3, 
                max_tokens=2000,
                stream=False
            )
        
            explanation = explanation_response.choices[0].message.content.strip()
            final_result = explanation
            return final_result
            
        except json.JSONDecodeError as e:
            return f"Ошибка парсинга JSON от LLM:\n{str(e)}\n\nПолученный ответ:\n{raw_response}\n\nОчищенный JSON:\n{clean_json_str}"
        except Exception as e:
            return f"Неожиданная ошибка:\n{str(e)}"

    def load_config(self):
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception as e:
                print("Ошибка загрузки конфига:", e)
                self.config = {}
        else:
            self.config = {}
    
    def save_config(self):
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({
                    "base_url": self.llm.base_url,
                    "model": self.llm.model,
                    "token": self.llm.token
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Ошибка сохранения конфига:", e)

    def get_chat_path(self, chat_id):
        return chats_dir / f"{chat_id}.json"

    def create_new_chat(self):
        if self.is_streaming:
            self.interrupt_stream(None)
        chat_id = str(uuid.uuid4())
        self.current_chat_id = chat_id
        self.clear_chat_ui()
        self.toggle_send_interrupt_button(streaming=False)
        return chat_id

    def delete_chat(self, chat_id):
        if not chat_id:
            return
        path = self.get_chat_path(chat_id)
        if path.exists():
            try:
                os.remove(path)
                if self.current_chat_id == chat_id:
                    self.create_new_chat()
                self.refresh_chat_list()
                chat_files = list(chats_dir.glob("*.json"))
                if not chat_files:
                    self.create_new_chat()
                    self.refresh_chat_list()
            except Exception as e:
                print(f"Ошибка удаления чата {chat_id}: {e}")

    def clear_chat_ui(self):
        self.chat_layout.clear_widgets()

    def load_chat(self, chat_id):
        if self.is_streaming:
            self.interrupt_stream(None)

        path = self.get_chat_path(chat_id)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                messages = json.load(f)
            self.current_chat_id = chat_id
            self.clear_chat_ui()
            for msg in messages:
                self.add_message(msg["text"], is_user=msg["is_user"], skip_save=True)
        except Exception as e:
            print("Ошибка загрузки чата:", e)
        self.refresh_chat_list()
        self.toggle_send_interrupt_button(streaming=False)

    def save_message_to_chat(self, text, is_user):
        if not self.current_chat_id:
            self.create_new_chat()

        path = self.get_chat_path(self.current_chat_id)
        messages = []
        file_existed_before = path.exists()

        if file_existed_before:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    messages = json.load(f)
            except:
                messages = []
        
        messages.append({"text": text, "is_user": is_user})
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            
            if not file_existed_before:
                Clock.schedule_once(lambda dt: self.refresh_chat_list(), 0)
                
        except Exception as e:
            print("Ошибка сохранения сообщения:", e)

    def save_message_to_chat_in_chat(self, text, is_user, chat_id):
        if not chat_id:
            return
        path = self.get_chat_path(chat_id)
        messages = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    messages = json.load(f)
            except:
                messages = []
        messages.append({"text": text, "is_user": is_user})
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Ошибка сохранения сообщения в чат:", e)

    def build(self):
        main_layout = BoxLayout(orientation='horizontal', padding=dp(10), spacing=dp(10))

        chat_panel = BoxLayout(orientation='vertical', size_hint_x=0.75)

        with chat_panel.canvas.before:
            Color(0.15, 0.15, 0.15, 1)
            self.chat_bg = Rectangle(size=chat_panel.size, pos=chat_panel.pos)
        chat_panel.bind(size=self._update_chat_bg, pos=self._update_chat_bg)
        
        top_bar = BoxLayout(size_hint_y=None, height=dp(40))
        top_bar.add_widget(Label(size_hint_x=1))
        self.settings_btn = Button(
            text='⚙',
            size_hint=(None, None),
            size=(dp(40), dp(40)),
            font_size=sp(20),
            background_color=(0.2, 0.6, 1, 1)
        )
        self.settings_btn.bind(on_press=self.open_settings)
        top_bar.add_widget(self.settings_btn)
        chat_panel.add_widget(top_bar)

        self.chat_scroll = NoDragScrollView(
            bar_width=dp(8),
            bar_color=(0.7, 0.7, 0.7, 0.8),
            bar_inactive_color=(0.7, 0.7, 0.7, 0.4),
            scroll_type=['bars', 'content'],
            do_scroll_x=False
        )
        self.chat_layout = BoxLayout(
            orientation='vertical', size_hint_y=None, spacing=dp(16),
            padding=[dp(12), dp(12)]
        )
        self.chat_layout.bind(minimum_height=self.chat_layout.setter('height'))
        self.chat_scroll.add_widget(self.chat_layout)
        chat_panel.add_widget(self.chat_scroll)

        input_panel = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(100))
        
        input_box = BoxLayout(size_hint=(1, None), height=dp(60), spacing=dp(8))
        self.user_input = FixedTextInput(
            hint_text='Введите сообщение...',
            multiline=True,
            font_size=sp(16),
            padding=[dp(12), dp(12)],
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            cursor_color=(0.2, 0.6, 1, 1)
        )
        self.user_input.bind(on_text_validate=self.send_message)
        send_btn = Button(
            text='\u25B6',
            size_hint_x=None, width=dp(60),
            font_size=sp(24),
            background_color=(0.2, 0.8, 0.3, 1)
        )
        send_btn.bind(on_press=self.send_message)
        input_box.add_widget(self.user_input, 1)
        input_box.add_widget(send_btn)
        self.send_btn = send_btn
        
        input_panel.add_widget(input_box)
        chat_panel.add_widget(input_panel)

        chat_list_panel = BoxLayout(orientation='vertical', size_hint_x=0.25)
        chat_list_label = Label(
            text='Чаты',
            size_hint_y=None,
            height=dp(30),
            font_size=sp(16),
            bold=True,
            color=(1, 1, 1, 1)
        )

        with chat_list_panel.canvas.before:
            Color(0.2, 0.2, 0.2, 1)
            self.list_bg = Rectangle(size=chat_list_panel.size, pos=chat_list_panel.pos)
        chat_list_panel.bind(size=self._update_list_bg, pos=self._update_list_bg)

        chat_list_panel.add_widget(chat_list_label)

        self.chat_list_scroll = ScrollView(do_scroll_x=False)
        self.chat_list_layout = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(4),
            padding=[dp(8), dp(8)]
        )
        self.chat_list_layout.bind(minimum_height=self.chat_list_layout.setter('height'))
        self.chat_list_scroll.add_widget(self.chat_list_layout)
        chat_list_panel.add_widget(self.chat_list_scroll)

        new_chat_btn = Button(
            text='+ Новый чат',
            size_hint_y=None,
            height=dp(40),
            background_color=(0.3, 0.7, 0.3, 1)
        )
        new_chat_btn.bind(on_press=self.create_and_load_new_chat)
        chat_list_panel.add_widget(new_chat_btn)

        main_layout.add_widget(chat_panel)
        main_layout.add_widget(chat_list_panel)

        Window.bind(on_key_down=self._on_key_down)
        Window.bind(on_key_up=self._on_key_up)
        Window.bind(on_resize=self._on_resize)
        Window.bind(on_textinput=self._on_textinput)

        Clock.schedule_interval(self.smart_scroll, 0.1)
        Clock.schedule_once(self._init_chat_width, 0.1)
        Clock.schedule_once(self.load_last_or_new_chat, 0.2)

        return main_layout

    def _update_list_bg(self, instance, value):
        self.list_bg.pos = instance.pos
        self.list_bg.size = instance.size

    def _update_chat_bg(self, instance, value):
        self.chat_bg.pos = instance.pos
        self.chat_bg.size = instance.size

    def create_and_load_new_chat(self, instance=None):
        self.create_new_chat()
        self.refresh_chat_list()

    def refresh_chat_list(self):
        self.chat_list_layout.clear_widgets()
        chat_files = sorted(chats_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        for chat_path in chat_files:
            chat_id = chat_path.stem
            title = self.get_chat_title(chat_id)
            chat_row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))

            title_label = Button(
                text=title,
                size_hint_x=0.85,
                halign='left',
                valign='middle',
                padding=[dp(8), dp(6)],
                background_color=(0.25, 0.25, 0.25, 1),
                color=(1, 1, 1, 1),
                font_size=sp(14),
                shorten=True,
                shorten_from='right'
            )
            title_label.bind(size=lambda instance, size: setattr(instance, 'text_size', (size[0], None)))
            title_label.bind(on_press=lambda btn, cid=chat_id: self.load_chat(cid))
            chat_row.add_widget(title_label)

            delete_btn = Button(
                text='✕',
                size_hint_x=0.15,
                background_color=(0.8, 0.3, 0.3, 1),
                font_size=sp(16),
                color=(1, 1, 1, 1)
            )
            delete_btn.bind(on_press=lambda btn, cid=chat_id: self.delete_chat(cid))
            chat_row.add_widget(delete_btn)

            self.chat_list_layout.add_widget(chat_row)

    def get_chat_title(self, chat_id):
        path = self.get_chat_path(chat_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                messages = json.load(f)
                if messages:
                    first_msg = messages[0]["text"]
                    if first_msg.startswith("Вы: "):
                        first_msg = first_msg[4:]
                    return first_msg[:15] + ("…" if len(first_msg) > 15 else "")
        except Exception as e:
            print(f"Ошибка чтения чата {chat_id}: {e}")
        return "Без названия"

    def load_last_or_new_chat(self, dt):
        chat_files = list(chats_dir.glob("*.json"))
        if chat_files:
            latest = max(chat_files, key=os.path.getmtime)
            chat_id = latest.stem
            self.load_chat(chat_id)
        else:
            self.create_new_chat()
        self.toggle_send_interrupt_button(streaming=False)

    def _init_chat_width(self, dt):
        self.refresh_all_labels(dt)

    def interrupt_stream(self, instance):
        if not self.is_streaming:
            return

        self.is_streaming = False

        if self.streaming_placeholder is not None and self.streaming_chat_id:
            label = self.streaming_placeholder
            if hasattr(label, 'message') and isinstance(label.message, str):
                if label.message.startswith("ИИ: ") and not label.message.endswith(" [Прервано]"):
                    new_text = label.text + " [Прервано]"
                    label.text = new_text
                    label.message = new_text
                    self.save_message_to_chat_in_chat(new_text, is_user=False, chat_id=self.streaming_chat_id)
            self.streaming_placeholder = None
            self.streaming_chat_id = None

        self.toggle_send_interrupt_button(streaming=False)

    def toggle_send_interrupt_button(self, streaming=False):
        if streaming:
            self.send_btn.text = '\u25A0'
            self.send_btn.background_color = (0.8, 0.2, 0.2, 1)
            self.send_btn.unbind(on_press=self.send_message)
            self.send_btn.bind(on_press=self.interrupt_stream)
        else:
            self.send_btn.text = '\u25B6'
            self.send_btn.background_color = (0.2, 0.8, 0.3, 1)
            self.send_btn.unbind(on_press=self.interrupt_stream)
            self.send_btn.bind(on_press=self.send_message)

    def _on_key_down(self, window, key, scancode, codepoint, modifier, *args):
        if key in (305, 306):
            self.ctrl_pressed = True

        def find_focused_textinput(widget):
            if isinstance(widget, TextInput) and widget.focus:
                return widget
            if hasattr(widget, 'children'):
                for child in widget.children:
                    result = find_focused_textinput(child)
                    if result:
                        return result
            return None

        focused_widget = find_focused_textinput(self.root)

        if not focused_widget:
            for child in Window.children:
                if child is not self.root:
                    focused_widget = find_focused_textinput(child)
                    if focused_widget:
                        break

        if focused_widget and 'ctrl' in modifier:
            if key in (ord('v'), KEY_M):
                focused_widget.paste()
                return True
            if key in (ord('a'), KEY_F):
                focused_widget.select_all()
                return True
            if key in (ord('z'), KEY_YA):
                focused_widget.do_undo()
                return True

        if self.user_input.focus and key == 13:
            if 'shift' in modifier:
                cursor = self.user_input.cursor_index()
                text = self.user_input.text
                self.user_input.text = text[:cursor] + '\n' + text[cursor:]
                Clock.schedule_once(lambda dt: self.user_input.do_cursor_movement('cursor_end'), 0)
                return True
            else:
                self.send_message(None)
                return True

        if self.focused_message and 'ctrl' in modifier:
            if key in (ord('c'), KEY_S):
                selected = self.focused_message.selection_text
                if selected:
                    Clipboard.copy(selected)
                return True
            if key in (ord('a'), KEY_F):
                self.focused_message.select_all()
                return True

        return False

    def _on_key_up(self, window, key, scancode, *args):
        if key in (305, 306):
            self.ctrl_pressed = False

    def _on_textinput(self, window, text):
        if self.ctrl_pressed:
            return True
        return False

    def _on_resize(self, window, width, height):
        Clock.schedule_once(self.refresh_all_labels, 0)

    @mainthread
    def refresh_all_labels(self, dt):
        available_width = max(200, self.chat_layout.width - dp(24))
        for child in self.chat_layout.children:
            if isinstance(child, SelectableLabel):
                child.update_width(available_width)

    def open_settings(self, instance):
        content = BoxLayout(orientation='vertical', spacing=dp(12), padding=dp(16))
        grid = GridLayout(cols=2, spacing=dp(10), size_hint_y=None, row_default_height=dp(48))
        grid.bind(minimum_height=grid.setter('height'))

        labels = ['Base URL:', 'Model:', 'Token:']
        inputs = [self.llm.base_url, self.llm.model, self.llm.token]
        self.setting_inputs = []

        for label, value in zip(labels, inputs):
            grid.add_widget(Label(text=label, font_size=sp(15), size_hint_x=None, width=dp(100)))
            inp = TextInput(text=value, multiline=False, font_size=sp(15), padding=[dp(8), dp(10)])
            self.setting_inputs.append(inp)
            grid.add_widget(inp)

        content.add_widget(grid)
        save_btn = Button(text='Сохранить', size_hint_y=None, height=dp(50), font_size=sp(16))
        save_btn.bind(on_press=self.save_settings)
        content.add_widget(save_btn)

        self.settings_popup = Popup(title='Настройки API', content=content, size_hint=(0.85, 0.7))
        self.settings_popup.open()

    def save_settings(self, instance):
        self.llm.base_url = self.setting_inputs[0].text.strip()
        self.llm.model = self.setting_inputs[1].text.strip()
        self.llm.token = self.setting_inputs[2].text.strip()
        self.llm.client = OpenAI(base_url=self.llm.base_url, api_key=self.llm.token)
        self.save_config()
        self.settings_popup.dismiss()

    def send_message(self, instance):
        text = self.user_input.text.strip()
        if not text:
            return

        self.save_message_to_chat(f"Вы: {text}", is_user=True)
        self.add_message(f"Вы: {text}", is_user=True, skip_save=True)
        self.user_input.text = ''

        placeholder = self.add_message("ИИ: ", is_user=False, skip_save=True)

        self.is_streaming = True
        self.streaming_placeholder = placeholder
        self.streaming_chat_id = self.current_chat_id
        self.response_buffer = ""
        self.last_update_time = 0

        self.toggle_send_interrupt_button(streaming=True)

        Thread(target=self.stream_response, args=(text, placeholder), daemon=True).start()

    def stream_response(self, query, label):
        full_text = "ИИ: "
        tools = [{
            "type": "function",
            "function": {
                "name": "solve_logic_task",
                "description": "Применяется, когда пользователь даёт логическую, математическую или текстовую задачу, требующую формализации сущностей и отношений для последующего логического вывода. Не используй для общих вопросов, объяснений или бесед.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_text": {
                            "type": "string",
                            "description": "Полный текст задачи, который нужно формализовать."
                        }
                    },
                    "required": ["task_text"]
                }
            }
        }]

        try:
            for event in self.llm.make_query_generator(query, tools=tools):
                if not self.is_streaming:
                    full_text += " [Прервано]"
                    break

                if event["type"] == "text":
                    full_text += event["content"]
                    current_time = time.time()
                    if current_time - self.last_update_time >= self.update_interval:
                        self.update_label(label, full_text)
                        self.last_update_time = current_time

                elif event["type"] == "tool_call":
                    args = json.loads(event["tool_call"]["function"]["arguments"])
                    tool_response = self.handle_tool_call(
                        event["tool_call"]["function"]["name"], args
                    )
                    full_text += "\n" + tool_response
            self.update_label(label, full_text, is_final=True)
            if self.streaming_chat_id:
                self.save_message_to_chat_in_chat(full_text, is_user=False, chat_id=self.streaming_chat_id)

        except Exception as e:
            error_text = f"ИИ: Ошибка: {e}"
            self.update_label(label, error_text, is_final=True)
            if self.streaming_chat_id:
                self.save_message_to_chat_in_chat(error_text, is_user=False, chat_id=self.streaming_chat_id)
        finally:
            self.is_streaming = False
            self.streaming_placeholder = None
            self.streaming_chat_id = None
            Clock.schedule_once(lambda dt: self.toggle_send_interrupt_button(streaming=False), 0)

    @mainthread
    def update_label(self, label, text, is_final=False):
        label.message = text
        label.text = text
        if is_final:
            label._update_height()

    def add_message(self, text, is_user=False, skip_save=False) -> SelectableLabel:
        if not skip_save and not (not is_user and text == "ИИ: "):
            self.save_message_to_chat(text, is_user)
        label = SelectableLabel(message=text)
        if is_user:
            label.foreground_color = (0.9, 0.95, 1, 1)
            label.background_color = (0.25, 0.25, 0.3, 1)
        else:
            label.foreground_color = (0.95, 0.95, 0.9, 1)
            label.background_color = (0.3, 0.3, 0.3, 1)

        available_width = max(200, self.chat_layout.width - dp(24))
        label.update_width(available_width)
        self.chat_layout.add_widget(label)
        return label

    def smart_scroll(self, dt):
        if not self.is_streaming:
            return
        if self.chat_scroll.scroll_y <= 0.05:
            self.chat_scroll.scroll_y = 0


if __name__ == '__main__':
    ChatApp().run()