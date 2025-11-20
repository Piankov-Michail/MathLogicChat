# MathLogicChat

<br>

## Run manually (Linux)
### Install dependencies:
```python
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
<br>

> For Windows instead `source env/bin/activate`: 
```python
env\Scripts\activate
```

### Run code:
```python
python -m src.main
```


### Run tests:
```python
pip install -r requirements-dev.txt
```
```python
pytest tests/ -v --tb=no 
```
<br>

> for more test details:

```python
pytest tests/ -v
```

<br>

## Build manually (get .bin, .exe) (Linux)
```python
source env/bin/activate
pip install pyinstaller
pyinstaller ChatApp_linux.spec
```
<br>

> For Windows:
```python
env\Scripts\activate
pip install pyinstaller
pyinstaller ChatApp_windows.spec
```

<br><br>
## [DEMO](https://docs.google.com/presentation/d/1JkvquSSn84EgfHs4uukGRbFLSOCg8MOxolrSS7R0HM8/edit?slide=id.p#slide=id.p)

<br><br>
## [Ready-made application](https://github.com/Piankov-Michail/MathLogicChat/releases/tag/v1.2.0)

<br><br>
## Working LLMs: 
### deepseek with tools support like: [this](https://build.nvidia.com/deepseek-ai/deepseek-v3_1-terminus)
### qwen3 with tools support like: [this](https://ollama.com/library/qwen3)



