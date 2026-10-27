# Датасет домашнего задания 1

**Студент:** Готовский Геннадий Дмитриевич, Б05-305.

**Датасет:** Amazon Reviews 2023 — Software.

**Первоисточник:** https://amazon-reviews-2023.github.io/

- [Отзывы Software](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Software.jsonl.gz).
- [Метаданные Software](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Software.jsonl.gz).

Взаимодействие — отзыв пользователя с оценкой товара (`parent_asin`); timestamp — время в миллисекундах Unix. Используются ID пользователей/товаров, оценки и метаданные. Релевантность: rating >= 4.

После дедупликации и iterative 5-core: 1 276 840 взаимодействий, 146 396 пользователей, 17 591 объект. Подготовка, временное разбиение, модели и результаты описаны в [experiments.ipynb](experiments.ipynb); инструкции запуска — в [README.md](README.md). Данные локальные, в Git не загружаются.
