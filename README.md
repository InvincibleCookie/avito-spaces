# Avito Spaces — восстановление пропусков пробелов (лёгкая локальная модель)

Лёгкий символьный **bi-GRU**-тэггер, который по строке **без пробелов** восстанавливает:
1) **позиции пропусков** (список индексов, где должен стоять пробел),
2) **сам текст** с расставленными пробелами.

> **Формат позиций (важно)**  
> Мы возвращаем **GAP-индексы** `j` — **индексы правых символов, перед которыми ставим пробел**.  
> Если пробел должен стоять между `s[i]` и `s[i+1]`, то **`j = i+1`** (индексация 0-based).

---

## Что уже включено в репозиторий

```
.
├─ README.md
├─ requirements.txt
├─ notebooks/
│  └─ Avito_DS_internship.ipynb          # ноутбук с полным пайплайном
├─ src/avito_spaces/
│  ├─ __init__.py
│  ├─ utils.py                            # clean_line, spaced→pair, конвертер индексов
│  ├─ data.py                             # сбор корпусов (WB/OSub/Genius) + пары (x,y)
│  ├─ model.py                            # SpaceTagger (bi-GRU) + Dataset/Collate/Vocab
│  ├─ train.py                            # обучение + калибровка порога + сохранение best.pt
│  └─ infer.py                            # загрузка модели, predict_phrase(), build_submission()
├─ scripts/
│  ├─ build_texts.py                      # собрать корпус train_texts.csv
│  └─ build_pairs.py                      # построить train_pairs.csv из train_texts.csv
└─ work/
   ├─ ckpt/
   │  └─ best.pt                          # ГОТОВЫЕ ВЕСА ДЛЯ ИНФЕРЕНСА
   └─ data/
      ├─ train_texts.csv                  # очищенные строки для обучения
      └─ train_pairs.csv                  # пары (x,y): «без пробелов» и таргеты
```

> Папка **`work/`** уже содержит веса и данные, поэтому **можно сразу запускать инференс и сборку submission**.

---

## Установка

Работает на Python **3.9–3.12**.

```bash
# (опционально) виртуальное окружение
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# зависимости (без torchvision/torchaudio — они не нужны)
pip install -r requirements.txt
```

Пакет `datasets` нужен **только** если будете пересобирать корпус (раздел «Обучение с нуля»).

---

## Быстрый старт (без обучения)

### 1) Единичный инференс: «фраза → пробелы»

```bash
python - << 'PY'
from avito_spaces.infer import predict_phrase
ckpt = "work/ckpt/best.pt"         # уже в репозитории
s = "книгавхорошемсостоянии"
spaced, gaps = predict_phrase(s, ckpt)
print("INPUT:", s)
print("SPACED:", spaced)           # строка с пробелами
print("GAPS:", gaps)               # список индексов j (ставим пробел ПЕРЕД s[j])
PY
```

### 2) Сбор `submission.csv` из исходного TXT

Входной `task_data` ровно как в условии: `id,text_no_spaces`.  
Парсер устойчив к запятым в тексте (используется «первый найденный» разделитель из `, \t ; |`).

```bash
python - << 'PY'
from avito_spaces.infer import build_submission
sub = build_submission(
    task_txt_path = "dataset_1937770_3.txt",  # ваш файл из задачи
    ckpt_path     = "work/ckpt/best.pt",
    out_csv_path  = "work/submission.csv"     # файл для загрузки в форму
)
print(sub.head())
print("Saved:", "work/submission.csv", "| shape:", sub.shape)
PY
```

Пример строк в готовом файле:

```
id,text_no_spaces,predicted_positions
0,куплюайфон14про,"[5, 12]"
1,ищудомвПодмосковье,"[3, 7, 11]"
...
```

Колонка **`predicted_positions`** — **строка** со списком **GAP-индексов**.

---

## Проверка формата перед отправкой

```bash
python - << 'PY'
import pandas as pd, re
df = pd.read_csv("work/submission.csv", dtype={"id":"int64","text_no_spaces":"string","predicted_positions":"string"})
assert {"id","text_no_spaces","predicted_positions"}.issubset(df.columns)
def ok(x):
    x = str(x).strip()
    if not (x.startswith("[") and x.endswith("]")): return False
    inner = x[1:-1].strip()
    if inner == "": return True
    return all(re.fullmatch(r"-?\d+", t.strip()) for t in inner.split(","))
assert df["predicted_positions"].map(ok).all()
print("submission.csv looks good:", df.shape)
PY
```

---

## Как это работает (кратко)

1. Из «правильных» строк строим пары `(x, y)`:
   - `x` — строка **без пробелов**,
   - `y[i] ∈ {0,1}` — должен ли стоять пробел **после** `x[i]`.
2. Обучаем **символьный bi-GRU** (emb → biGRU → линейный слой).
3. На инференсе переводим «пробел **после i**» → **GAP-индекс** `j = i+1` (**перед** `s[j]`) и применяем простые пост-правила:
   - не разделять **цифру от цифры**;
   - не ставить перед правой пунктуацией `.,!?:;…)]}»`;
   - не ставить сразу после левой пунктуации `([{«`.
4. Порог вероятности калибруем на валидации по F1 и сохраняем в `best.pt`.

---

## Обучение на готовых парах (включено)

Если хотите переобучить на уже подготовленных парах:

```bash
python - << 'PY'
from pathlib import Path
from avito_spaces.train import train_main
ckpt, thr = train_main(
    pairs_csv    = Path("work/data/train_pairs.csv"),
    work_dir     = Path("work"),
    epochs       = 10,        # увеличьте для лучшего качества
    batch        = 256,       # уменьшайте на CPU
    emb          = 96,
    hidden       = 192,
    lr           = 3e-3,
    use_cpu_only = False
)
print("Saved:", ckpt, "| best_thr:", thr)
PY
```

Новый чекпоинт перезапишет `work/ckpt/best.pt`.

---

## Обучение с нуля (опционально)

Эти шаги **не обязательны**, если вы используете уже включённые веса.

1) **Собрать корпус** (Wildberries sample, OpenSubtitles RU, HF Genius RU):

```bash
python -m scripts.build_texts
# создаст work/data/train_texts.csv
```

2) **Построить пары** `(x,y)`:

```bash
python -m scripts.build_pairs
# создаст work/data/train_pairs.csv
```

3) **Обучить модель** (см. раздел выше).

> Для шага (1) может понадобиться `datasets==2.19.0`.  
> `torchvision/torchaudio` не требуются.

---

## Пример быстрого теста

```bash
python - << 'PY'
from avito_spaces.infer import predict_phrase
ckpt = "work/ckpt/best.pt"
tests = [
    "книгавхорошемсостоянии",
    "ищуквартирууметро",
    "новаямикроволновкаSamsung",
    "куплюайфон14про",
]
for s in tests:
    spaced, gaps = predict_phrase(s, ckpt)
    print(f"{s} -> {spaced}   {gaps}")
PY
```

---

## Ресурсы

- **CPU-режим**: ставьте `use_cpu_only=True` и уменьшайте `batch` (64–128).
- **GPU (Colab)**: достаточно T4/16GB; параметры по умолчанию подходят.

---

## Соответствие правилам задачи

- Локальная компактная модель (**без внешних API/LLM**).
- Полная воспроизводимость: код/ноутбук + `work/` с весами.
- Метрика: `predicted_positions` — строка со списком **GAP-индексов** (формат из условия).
- Источники данных: открытые корпуса (WB sample, OpenSubtitles RU, HF Genius RU) указаны.

---

## Лицензии (данные)

- **Wildberries dataset sample** — открытая выборка названий товаров.
- **OpenSubtitles RU (OPUS)** — публичные русскоязычные фразы.
- **Genius RU lyrics** — `sevenreasons/genius-lyrics-russian` (HuggingFace).

Используются для обучения общей модели восстановления пробелов.

---
