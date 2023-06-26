---
layout: post
title: Слабое обучение с учителем (weakly supervised learning)
date: 2023-05-01 19:51 +0300
img_path: /assets/post/weakly-supervised
categories: [Tutorial]
tags: [weak supervision, unsupervised]
---

Машинное обучение произвело революцию в способах решения сложных задач, от обработки естественного языка до распознавания изображений.
Однако одно из основных препятствий, с которыми сталкиваются специалисты по машинному обучению,
является недостаточное количество размеченных данных, которые часто необходимы для качественного обучения моделей.
Маркирование же всех данных обычно либо слишком дорого, либо не всегда возможно.
К счастью, слабое обучение с учителем стало мощным решением этой проблемы.
В этой статье мы рассмотрим три типа данного обучения и методы, используемые в каждом из них.

![Unlabeled data](weakly supervided learning.jpg)
_Пример влияния неразмеченных данных в слабом обучении с учителем (weakly supervised learning). [Источник](https://en.wikipedia.org/wiki/Weak_supervision)_

Три типа слабого обучение с учителем:
* Неполное обучение - размечено только небольшое подмножество тренировочных данных, тогда как остальные данные остаются не размеченными;
* Приблизительное обучение - даны только приблизительные метки;
* Неточное обучение - данные метки не всегда соответствуют действительности.

Теперь, вкратце остановимся на каждом типе обучения и методах, используемые для их решения.

## Неполное обучение

Активное обучение (active learning) и частичное обучение с учителем (semi-supervised learning) — два основных метода,
используемых при неполном обучении. Активное обучение предполагает, что существует «оракул», который обладает экспертной оценкой,
к которому можно обратиться для получения правильных меток.
С другой стороны, частичное обучение с учителем пытается использовать неразмеченные данные в дополнение к размеченным данным
для повышения эффективности обучения без какого-либо вмешательства человека.

Основная цель активного обучения — минимизировать количество запросов для снижения стоимости обучения.
Эту проблему можно решить, попытавшись выбрать наиболее ценные немаркированные экземпляры для запроса,
используя два критерия: информативность и репрезентативность.

Информативность описывает, насколько хорошо немаркированный экземпляр помогает уменьшить неопределенность статистической модели,
а репрезентативность описывает, насколько хорошо экземпляр помогает представить структуру входных шаблонов.

В частичном обучении с учителем не участвует человек с экспертной оценкой, и алгоритм пытается исследовать данные,
используя методы обучения без учителя, такие как кластерные и множественные предположения.
Оба предположения основываются на том, что похожие точки данных должны иметь похожие выходные данные.

Где-то между этими двумя методами есть еще один, который смешивает оба подхода.
В этом методе есть функции маркировки, которые даны экспертами. Эти функции охватывают некоторую часть данных корпуса.
Используя эти размеченные точки данных, мы можем обучить вероятностную модель для маркировки других точек, которые небыли размечены функциями.
Такие решения, как [Snorkel](https://snorkel.ai/) от Stanford, [skweak](https://spacy.io/universe/project/skweak) для обработки языка и [ASTRA](https://github.com/microsoft/ASTRA) от Microsoft используют данный подход.

Стоит так же отметить, что, хотя ожидается, что эффективность обучения улучшится за счет использования неразмеченных данных,
в некоторых случаях производительность может ухудшиться после частисного обучения с учителем.
Использование неразмеченных данных естественным образом приводит к более чем одному варианту модели,
а неправильный выбор может привести к снижению производительности.
Основная стратегия сделать частичное обучение с учителем «более безопасным» состоит в том,
чтобы оптимизировать наихудшую результативность среди вариантов, путем включения механизмов ансамбля.

## Приблизительное обучение

Мультиэкземплярное обучение (multi-instance learning) является основным подходом, используемым в приблизительном обучении.
В мультиэкземплярном обучении множество точек данных является положительным, если некоторое подмножество также является положительным.
Цель мультиэкземплярного обучения состоит в том, чтобы предсказать метки для неизвестных множеств.
Этот тип обучения обычно используется в таких приложениях, как фармацевтическое исследование лекарств и анализ медицинских изображений.

## Неточное обучение

Основная идея неточного обучения состоит в том, чтобы выявить потенциально ошибочно размеченные данные и внести исправления.
Этого можно достичь, используя стратегии голосования или методы кластеризации для поиска выбросов.
Выявляя и исправляя неправильно помеченные примеры, можно улучшить качество тренировочных данных и, следовательно, точность моделей.

## Заключениие
Можно сказать, что слабое обучение с учителем стало мощным решением проблемы высокой стоимости маркировки данных.
На практике решение обычно включает сочетание всех трех типов обучения.