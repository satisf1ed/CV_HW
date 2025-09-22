# CV_HW

Логи и профилировку можно скачать по ссылке: https://disk.360.yandex.ru/d/jSpaf40wI6HFeA

## Запуск

1. Скачайте .ipynb файл и откройте его в Google colab. (или откройте этот же файл напрямую по ссылке: https://colab.research.google.com/drive/19tyHf7UsphEVO_IAjE76QlkEBtm0tN8m?usp=sharing)
2. Выставите Runtime -> Change Runtime Type -> T4 GPU перед запуском
3. Запускайте ячейки по очереди до конца

## Описание Эксперимента

Этот репозиторий содержит полностью готовый к запуску в Google Colab эксперимент по сравнению простой CNN и линейного классификатора (linear probe) поверх предобученного ViT-Tiny. Включены: подготовка данных, sanity-check (оверфит нескольких батчей), логирование в TensorBoard, профилировка torch.profiler, расчёт метрик (accuracy, macro-F1), построение матриц ошибок и сохранение чекпоинтов/отчёта.

### Данные

Автосборка мини-датасета из CIFAR-10
Выбираются ≥5 классов, фиксируется количество изображений на класс, и всё сохраняется в структуре ImageFolder.

Аугментации:
- train: RandomResizedCrop(224, scale=(0.8,1.0)), RandomHorizontalFlip, нормализация под ImageNet.

- val: Resize(256) → CenterCrop(224), нормализация под ImageNet.


### Архитектуры
1. Простая CNN
  3 свёрточных блока с Conv2d + BatchNorm2d + ReLU + MaxPool2d, затем AdaptiveAvgPool2d → Linear классификатор. Размер входа — 224×224.

2. ViT-Tiny (linear probe)
  Модель vit_tiny_patch16_224 из timm с предобученными весами:
  - Заморожен весь бэкон (включая режим eval()).
  - Обучается только линейная голова (head.train()).

### Обучение

В тетрадке реализован общий цикл обучения:
- AdamW, AMP (autocast + GradScaler), Cosine LR со стадийным warmup.
- Логирование в TensorBoard: train/val loss, accuracy, macro-F1, LR.
- Гистограммы весов и градиентов.
- Sanity-check: отдельный прогон для оверфита на 1–2 батчах — должен быстро достигать acc ≈ 1.0.

### Профилировка

Используется torch.profiler с расписанием wait 5 / warmup 5 / active 50–100 шагов. Трейс сохраняется в подкаталог TensorBoard-логов (.../profiler) и просматривается в TensorBoard → Profile.

### Оценка и визуализация

- Метрики на val: accuracy, macro-F1.
- Classification report по классам.
- Confusion matrix в двух видах: абсолютные значения и нормированная по строкам. PNG-файлы сохраняются в runs/....

Также сохраняются чекпоинты:

- checkpoints/cnn_best.pt
- checkpoints/vit_tiny_linear_probe.pt

И JSON-отчёт-шаблон:
- runs/.../final_report.json с полями: пути к логам/чекпоинтам, список классов, заметки и итоговые выводы.
