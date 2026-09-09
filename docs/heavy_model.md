# Heavy: отчёт о реализации и запуске

Реализован `siam_mit_b3_diff`: один общий MiT-B3 encoder, абсолютные разности
четырёх настоящих уровней признаков, свёрточный top-down decoder и один сырой
логит на пиксель. Полная глубина по умолчанию — **(3, 4, 18, 3), 28 блоков**.
Это проектная архитектура, а не точная реализация ChangeFormerV6.

Проверка выполнена 8 сентября 2026 года в существующем окружении:
Python 3.11.9, PyTorch 2.13.0+cpu, torchvision 0.28.0+cpu, Windows.
CUDA недоступна, видимых GPU — 0, Gloo доступен. Зависимости не устанавливались.

**Реализовано и проверено:** архитектура, численная проверка attention, полные
градиенты и AdamW step, factory, shared fit/evaluation/checkpoints/resume,
HPO resolution, preprocessing, checkpointing с сохранением RNG, два CPU-процесса
Gloo и безопасные notebooks. Итог: **68 passed, 2 skipped, 1 warning**.

**Реализовано, но GPU-исполнение не проверено:** CUDA BF16, NCCL/DDP на A100,
1024×1024 forward/backward, измерение VRAM и времени CUDA-step, независимые HPO workers.
Два CUDA-теста пропущены именно из-за отсутствия CUDA и явного выделения устройства.

Не осталось намеренно отложенных пунктов реализации внутри согласованного объёма.
Полное обучение Heavy, полный HPO, подготовка/скачивание LEVIR-CD и финальная
исследовательская оценка test **не запускались**. Измерений VRAM, GPU throughput
или качества обученной Heavy-модели нет.

## Изменённые файлы

Пути ниже отсчитываются от
`E:/Projects/Satellite change intelligence/satellite-change-intelligence`.

| Файл | Изменение |
|---|---|
| `src/satchange/models/siam_mit_b3_diff.py` | Полный encoder, SR attention, Mix-FFN, stochastic depth, decoder, проверки входов и activation checkpointing |
| `src/satchange/models/factory.py` | Регистрация `siam_mit_b3_diff` |
| `src/satchange/models/__init__.py` | Публичный экспорт `SiamMiTB3Diff` |
| `src/satchange/execution.py` | Общие FP32/BF16, seed/RNG, rank-zero coordination, unwrap и сбор небольших метаданных |
| `src/satchange/training.py` | Опциональная precision, DDP validation/checkpoint coordination, RNG resume, per-device memory metadata |
| `src/satchange/optimization.py` | Сохранено `model__` routing; проверки конфликтов/неподдерживаемых параметров, запрет nested DDP, precision и metadata, защита существующих search directories |
| `src/satchange/run.py` | Общий YAML launcher, torchrun, reconstruction, запуск существующего HPO и ограниченный CUDA probe |
| `configs/siam_mit_b3_diff_resized.yaml` | Whole-scene resize 256, полный B3, scratch FP32 |
| `configs/siam_mit_b3_diff_full_resolution.yaml` | Native `image_size: null`, полный B3, checkpointing включён |
| `configs/siam_mit_b3_diff_search.json` | Исполняемый пример HPO с архитектурными и связанными loss-кандидатами |
| `notebooks/03_heavy.ipynb` | Безопасный CPU smoke и guarded experiment lifecycle |
| `notebooks/02_bit.ipynb` | Seed внутри tiny-overfit guard, уникальная директория, подготовка данных остаётся в отдельном data guard |
| `tests/test_heavy.py` | Conformance, attention reference, gradients, fit/resume/HPO, opt-in CUDA probes |
| `tests/test_execution.py` | Gloo coordination, preprocessing, launcher и Light/BIT checkpoint regressions |
| `tests/test_notebooks.py` | nbformat, IPython-aware syntax и свежий kernel Heavy из `notebooks/` |
| `docs/heavy_model.md` | Этот отчёт и команды запуска |

Light/BIT модели, их численные конфигурации, historical artifacts, datasets и
старые notebook outputs не изменены. Репозиторий перед работой был чистым.

## Фактический поток тензоров

Следующие формы получены hooks на **default-depth forward 256×256**, B=1.
Encoder действительно вызван дважды как один и тот же объект.

| Узел | Форма для общего B | Q/KV tokens | Heads × head_dim |
|---|---|---|---|
| A, B | `[B,3,256,256]` | — | — |
| F1, depth 3 | `[B,64,64,64]` | 4096 / 64 | 1 × 64 |
| F2, depth 4 | `[B,128,32,32]` | 1024 / 64 | 2 × 64 |
| F3, depth 18 | `[B,320,16,16]` | 256 / 64 | 5 × 64 |
| F4, depth 3 | `[B,512,8,8]` | 64 / 64 | 8 × 64 |
| T1 | `[B,128,64,64]` | — | — |
| T2 | `[B,128,32,32]` | — | — |
| T3 | `[B,128,16,16]` | — | — |
| T4 | `[B,128,8,8]` | — | — |
| U4 → U3 → U2 → U1 | `[B,128,8,8]` → `[B,128,16,16]` → `[B,128,32,32]` → `[B,128,64,64]` | — | — |
| Half-resolution refinement | `[B,64,128,128]` | — | — |
| Full-resolution refinement | `[B,32,256,256]` | — | — |
| Logits | `[B,1,256,256]` | — | — |

Каждая T-проекция получает `abs(F_before - F_after)` до Conv/GroupNorm/GELU.
U4 содержит один ConvBlock; каждая из трёх fusion содержит два ConvBlock.
Refinement содержит по одному ConvBlock на половинном и полном разрешении.
Интерполяции — bilinear, `align_corners=False`, с явными целевыми размерами.

Для native 1024 encoder формы аналитически составляют
`[B,64,256,256]`, `[B,128,128,128]`, `[B,320,64,64]`, `[B,512,32,32]`.
Это **вывод из stride**, а не выполненный native GPU-тест.

Из `parameters()` фактически получены:

| Часть | Параметры |
|---|---:|
| Один encoder | 44 072 128 |
| Проекции + decoder + refinement + head | 1 700 833 |
| Всего | 45 772 961 |
| Trainable | 45 772 961 |

Общий encoder считается один раз, хотя вычисляется дважды. Dummy/classification
параметров нет. Оценка около 45.8M из задания согласуется с точным подсчётом.

## Численная реализация и происхождение

Overlap embedding: biased Conv 7/4/3, затем три Conv 3/2/1. Transformer и выходные
LayerNorm используют eps=1e-6; embedding/SR LayerNorm — eps=1e-5.
SR ratios 8/4/2/1, Q не сокращаются. SR convolution обычная, groups=1.
На последнем уровне не создаются неиспользуемые SR convolution/norm.

Mix-FFN: Linear C→4C, spatial depthwise 3×3 с groups=4C, GELU, dropout,
Linear 4C→C, dropout. Pre-norm residual branches используют независимые вызовы
DropPath; вероятность линейно растёт по всем 28 блокам.

Инициализация один раз: Linear truncated normal std=0.02, Conv normal
std=sqrt(2/fan_out), где fan_out учитывает groups; normalization weight=1,
bias=0. Encoder не содержит positional tables, classification tokens или heads.

Операции написаны независимо по заданию с проверкой
[официального MiT reference](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py).
Файлы стороннего проекта не копировались и не vendored. Архитектурный контекст:
[SegFormer](https://arxiv.org/abs/2105.15203) и
[ChangeFormer](https://github.com/wgcban/ChangeFormer).

SDPA заменяет явное построение attention matrix; возможны обычные различия
округления между backend implementations. Tiny double-precision reference
проверяет outputs, input gradients и parameter gradients для SR=1 и SR=2.
В eval attention dropout всегда 0. CUDA backend не форсируется; Flash Attention
не заявляется. Использован контракт установленного PyTorch 2.13, проверенный
через локальные SDPA docstring и `inspect.signature(checkpoint)`; версия 2.13
внешней API-страницы не открылась. Общая справка:
[checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html),
[DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).

Checkpointing вызывает реальные encoder blocks с явными H/W,
`use_reentrant=False`, `preserve_rng_state=True`. Работает только в training
с включёнными градиентами, без требования `requires_grad` у dataset tensors.
Output/gradient agreement проверено также со stochastic depth/dropout.

`pretrained_backbone=True` явно вызывает NotImplementedError: внешний pretraining
требует отдельной работы с provenance и normalization. Scratch не скачивает weights.

## Переиспользованная инфраструктура и preprocessing

Оставлены общие LEVIR-CD dataset, paired transforms, BCEDiceLoss,
BinaryChangeMetrics, train_one_epoch, evaluate_thresholds, fit_binary_change_model,
factory, `_create_scheduler`, `_expand_search_space`, `_build_trial_parameters`
и `run_parameter_search`. Отдельного Heavy trainer/HPO/dataset нет.
`load_config` уже generic, поэтому FC-Siam-specific experiment wrapper не менялся.

На синтетической сцене 1024×1024 с двумя угловыми областями проверено:
resized выдаёт весь scene в 256×256, native сохраняет 1024×1024;
RGB использует существующий bilinear, mask nearest-neighbor и остаётся бинарной.
Обе временные картинки и маска получают согласованные contiguous transforms.

Конфиги используют одинаковую полную архитектуру. Native включает checkpointing.
Batch size 1 — непроверенная консервативная стартовая величина. AdamW lr=1e-4,
weight_decay=1e-2, BCE/Dice=0.3/0.7, pos_weight=10, seed=42, 100 epochs —
предварительный recipe будущего запуска. pos_weight=10 записан явно и не выдаётся
за заново рассчитанную class balance. Scheduler — none; FP32 по умолчанию.

Threshold grid сохранён: 0.05…0.95 с шагом 0.05. Счётчики накапливаются глобально,
sigmoid вычисляется один раз на validation batch, выбран максимум validation F1,
при равенстве — первый threshold. F1 rank/batch averages не используются.
Существующее поведение нулевых знаменателей и probability diagnostics сохранено.

## Минимальный путь YAML → model → forward

Из корня репозитория, в установленном окружении проекта:

```python
import torch
from satchange.utils import load_config
from satchange.run import resolve_parameters
from satchange.models import create_change_model

config = load_config("configs/siam_mit_b3_diff_resized.yaml")
parameters = resolve_parameters(config)
model = create_change_model(config["model"]["name"], parameters["model_parameters"]).eval()
with torch.no_grad():
    logits = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
assert logits.shape == (1, 1, 64, 64)
```

Default-depth common-fit test также действительно выполнил loss/backward/AdamW,
проверил конечные ненулевые градиенты **каждого weight tensor** и их изменение,
best/last/history, strict reconstruction, evaluation и ещё одну resume epoch.
Обе временные ветви получают input gradients в отдельном тесте.

## Команды будущих запусков — не исполнялись

Из корня репозитория, после активации окружения с CUDA PyTorch и выделения GPU.
`--device` задаёт логический индекс внутри существующего CUDA_VISIBLE_DEVICES;
launcher не переопределяет allocation. Подготовленный LEVIR-CD должен существовать.
Уникальное имя run генерируется автоматически, существующие outputs не перезаписываются.

Одна GPU, resized:

```bash
python -m satchange.run train --config configs/siam_mit_b3_diff_resized.yaml --device cuda:0
```

Одна GPU, native и явно выбранный BF16:

```bash
python -m satchange.run train --config configs/siam_mit_b3_diff_full_resolution.yaml --device cuda:0 --precision bf16
```

Четыре GPU, один DDP trial:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 -m satchange.run train --config configs/siam_mit_b3_diff_resized.yaml --device cuda --precision bf16 --ddp
```

Четыре независимых HPO workers:

```bash
python -m satchange.run hpo --config configs/siam_mit_b3_diff_resized.yaml --gpu-ids 0 1 2 3 --search-space configs/siam_mit_b3_diff_search.json --precision bf16
```

JSON задаёт 32 комбинации, включая два depth-кандидата; это существенный
будущий эксперимент, не smoke. Чистое CPU-разрешение параметров можно проверить так:

```python
from satchange.optimization import _build_trial_parameters
trial = _build_trial_parameters(parameters, {
    "model__encoder_depths": (3, 4, 6, 3),
    "model__decoder_channels": 64,
    "learning_rate": 3e-4,
    "loss_weights": (0.3, 0.7),
})
assert trial["model_parameters"]["encoder_depths"] == (3, 4, 6, 3)
```

Последовательность depths — один candidate, не четыре независимых axes.
Nested mappings глубоко копируются, `model__` удаляются после разрешения,
linked loss_weights раскладываются в BCE/Dice. Non-default depths получают
label `depth_ablation`, реальные parameter counts и полную resolved configuration.
HPO сохраняет отдельные model instances/optimizers/loaders на каждой GPU.
Внутри torchrun или активной process group запуск HPO отклоняется.

## DDP и resume

DDP использует LOCAL_RANK/RANK/WORLD_SIZE; CUDA backend — NCCL, CPU — Gloo.
Process group уничтожается в finally. Один model replica на процесс,
`find_unused_parameters=False`. Математический train loop общий.

`data.batch_size` — число пар **на процесс**; global batch = batch × world_size.
Training DistributedSampler имеет `drop_last=False`: дополняет выборку до равных
длин ranks, при необходимости повторяя небольшое число training samples.
На каждой эпохе вызывается set_epoch. Seed до model construction одинаковый;
augmentation и loader RNG streams затем rank-aware.

Rank 0 оценивает полный validation loader без DistributedSampler и без DDP forward.
Остальные ranks получают broadcast результата или сообщения об ошибке. Scheduler,
threshold и фиксированное окончание epoch budget согласованы. Новый early-stopping
алгоритм не добавлялся: его не было в исходном trainer. Только rank 0 сохраняет
общие checkpoints. Gloo test проверяет 3 validation samples без padding duplicates,
одинаковые weights, plateau scheduler, resume и распространение validation failure.

Старые BatchNorm слои не превращаются в SyncBatchNorm; для Light/BIT остаётся
локальная статистика. Текущий DiceLoss считает Dice отдельно по sample, затем mean;
DDP усредняет rank-local loss gradients. Эквивалентность иной global-batch Dice
формулировке или битовая идентичность одной/четырём GPU не обещается.

Checkpoint содержит unwrapped state_dict без `module.`, optimizer, scheduler,
epoch, history, best/current metrics, config, precision, world_size и rank RNG states.
Reconstruction из доверенного проекта: `reconstruct_checkpoint(path, device)`.
Legacy nested YAML и flat HPO metadata поддерживаются; strict state loading обязателен.

Resume требует исходное имя run, ту же architecture, precision и world_size.
`--epochs` при resume означает **число дополнительных эпох**, как в исходном fit.
Рекомендуется передать тот же YAML, dataset, batch/worker settings и threshold grid.
RNG восстанавливается на границе эпохи; mid-epoch и persistent-worker state replay
не поддерживаются. Старые checkpoints без RNG по-прежнему загружаются, но не дают
новой гарантии точного RNG continuation. Изменение world size отклоняется.

Начать новый опыт с весами checkpoint — отдельный вызов reconstruction и новый
optimizer/run; это не resume. Test cell использует единственный threshold из
сохранённых validation metrics выбранного checkpoint и не подбирает его на test.

## Измерение GPU

```bash
python -m satchange.run probe --config configs/siam_mit_b3_diff_resized.yaml --device cuda:0 --precision bf16 --enable-probe
python -m satchange.run probe --config configs/siam_mit_b3_diff_full_resolution.yaml --device cuda:0 --precision bf16 --enable-probe --allow-native-probe
```

Probe выполняет один реальный BCE+Dice/AdamW step со свежим model и optimizer;
step включает создание AdamW state. Измеряется синхронизированное время, включая
передачу synthetic batch, peak allocated/reserved MiB, device/GPU/version/config,
precision, checkpointing и параметры. Результат сохраняется в уникальный
`artifacts/profiling/<uuid>.json`. OOM записывается как failure исходной конфигурации;
разрешение/глубина автоматически не уменьшаются. GPU не сбрасывается.

Resized разрешает один выбранный batch из 1/2/4; автоматического поиска предела нет.
Native только batch 1 с дополнительным opt-in. В DDP memory history сохраняет
per-rank peaks и maximum per-device peak, не сумму как «доступную одной GPU».
GPU-hours учитывают число процессов ровно один раз. DDP не объединяет VRAM четырёх
устройств в единое адресное пространство. Измерения здесь не выполнялись.

Opt-in GPU regression tests используют `SATCHANGE_TEST_CUDA_DEVICE=cuda:0`;
native дополнительно требует `SATCHANGE_TEST_NATIVE=1`. Обычный pytest не занимает GPU.

## Проверки и notebooks

Команды реально запускались через существующий `.venv/Scripts/python`:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m pytest tests/test_heavy.py
.\.venv\Scripts\python -m pytest tests/test_execution.py
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\python -m ruff check . --output-format concise
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m satchange.run --help
git diff --check
git -c core.whitespace=cr-at-eol diff --check
```

Полный итоговый pytest: **68 passed, 2 skipped, 1 warning, 39.39 s**.
До изменений: 38 passed. Первоначальные focused Heavy tests: 23 passed;
первоначальные runtime tests: 4 passed; дополнительные legacy/notebook tests
включены в полный итог. Два новых GPU probes пропущены. Warning — существующие
notebook cells без id; старые notebooks не переформатированы ради удаления warning.

Focused Ruff и mypy:

```powershell
.\.venv\Scripts\python -m ruff check src/satchange/models/siam_mit_b3_diff.py src/satchange/models/factory.py src/satchange/models/__init__.py src/satchange/execution.py src/satchange/run.py src/satchange/training.py src/satchange/optimization.py tests/test_heavy.py tests/test_execution.py tests/test_notebooks.py notebooks/03_heavy.ipynb
.\.venv\Scripts\python -m mypy src/satchange/models/siam_mit_b3_diff.py src/satchange/models/factory.py src/satchange/models/__init__.py src/satchange/execution.py src/satchange/run.py src/satchange/training.py src/satchange/optimization.py
```

Оба прошли: Ruff — All checks passed; mypy — no issues in 7 source files.
compileall и CLI help прошли. Обычный diff --check отмечает CRLF добавленных строк
в исходно CRLF-tracked training.py; проверка с `cr-at-eol` прошла.
Исходный формат строк сохранён, массовая нормализация файла не выполнялась.

Repository-wide Ruff: **17 существующих ошибок** (было 18; исправлен import order
в изменённом optimization.py). Repository-wide mypy: **те же 14 ошибок в 5 файлах**.
Это не новые Heavy regressions. В частности, остались `download_dorectory`,
неопределённый DataLoader в незавершённом helper, дефекты dormant blocks.py,
старые no-any-return/type annotations, line length/import/zip diagnostics.
Они не исправлялись вне объёма задачи.

| Notebook | Проверено | Не запускалось |
|---|---|---|
| `01_fc_siam_diff.ipynb` | nbformat, IPython-aware compilation, полный safe fresh-kernel execution на CPU; чтение существующих summaries | Data preparation и HPO guards |
| `02_fc_siam_diff_full_hpo.ipynb` | То же, включая чтение существующих summaries | Native HPO/data guards |
| `02_bit.ipynb` | То же, CPU factory и synthetic forward | Подготовка данных/tiny-overfit |
| `03_heavy.ipynb` | То же, полный default-depth CPU forward; запуск из root и отдельно из notebooks/ | Data, CUDA probe, tiny-overfit, training, HPO, checkpoint inspection, final test |

Проверки legacy notebooks выполнялись отдельным nbclient script с новым kernel
на `.venv` interpreter для каждого файла. Heavy повторно выполнен автоматическим
`test_notebooks.py` из каталога notebooks/. Исполненные outputs не сохранялись
в пользовательские notebooks. Никакие production checkpoints не загружались.

Единственные практические ограничения приёмки — отсутствие CUDA/NCCL/A100 и
настоящих исследовательских запусков. Заданная архитектура не упрощалась ради CPU:
shallow depths используются только в явно отмеченных изолированных тестах и HPO ablations.
