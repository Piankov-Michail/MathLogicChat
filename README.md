# MathLogicChat

<br>

## Run manually
### Install dependencies:
```python
python3 -m venv env
source env/bin/activate
pip install -r requerements.txt
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

## Build manually (get .exe, .bin)
```python
source env/bin/activate
pip install pyinstaller
pyinstaller ChatApp.spec
```
