# День 03 — Глубокое обучение
## Глубокое обучение
Знакомство с моделями Text to speech, Object Detection, Text Summarization и их ограничениями.

## Оглавление
1. [Глава I](#глава-i) \
    1.1. [Преамбула](#преамбула)
2. [Глава II](#глава-ii) \
    2.1. [Общая инструкция](#общая-инструкция)
3. [Глава III](#глава-iii) \
    3.1. [Цели](#цели)
4. [Глава IV](#глава-iv) \
    4.1. [Задание](#задание)
5. [Глава V](#глава-v) \
    5.1. [Сдача работы и проверка](#сдача-работы-и-проверка)

## Глава I
### Преамбула

Глубокие нейронные сети — это мощный инструмент машинного обучения, позволяющий моделировать сложные зависимости между входными данными и выходными результатами. Они состоят из множества слоев, каждый из которых обрабатывает информацию на разных уровнях абстракции. Благодаря этому глубокие нейронные сети могут обрабатывать большие объемы данных и достигать высокой точности в различных задачах, таких как распознавание изображений, обработка естественного языка и прогнозирование временных рядов. Однако обучение глубоких нейронных сетей требует больших вычислительных ресурсов и может быть сложным процессом, требующим тщательной настройки параметров модели.

<details><summary>Расписание проектов</summary>

**День 00. Сбор данных**: \
Данные находятся в разрозненных источниках. Их надо собрать и объединить в единый датасет и разобраться, какие данные у нас есть.

**День 01. Анализ данных:** \
Работа с простыми дескриптивными статистиками. Цель — чуть лучше понять анализ данных, выявить дополнительные проблемы с качеством данных и решить их. Ты построишь гистограммы, разные графики, чтобы еще лучше понять, как устроены данные.
Всё это может дать идеи для создания новых продуктов.

**День 02. Машинное обучение:** \
Займемся задачей предсказания оттока клиентов. Подготовим данные к обучению.
Обучим модель машинного обучения. Оценим качество предсказания.

**День 03. Глубокое обучение ← Ты находишься здесь:** \
Познакомимся с несколькими моделями глубокого обучения. 

**День 04. Внедрение:** \
Используя искусственный интеллект, ты разработаешь идею улучшения существующего процесса в своей сфере. Ты оценишь финансовый эффект от модели, трудозатраты, необходимые ресурсы, какой точности нужно добиться и с кем из стейкхолдеров следует переговорить.
</details>

## Глава II
### Общая инструкция

Методология «Школы 21» может быть не похожа на тот образовательный опыт, который с тобой случался ранее. Ее отличает высокий уровень автономии: у тебя есть задача, и ты должен ее выполнить. По большей части тебе нужно будет самому добывать знания для ее решения. Второй важный момент — это peer-to-peer обучение. В образовательном процессе нет преподавателей и экспертов, перед которыми ты защищаешь свой результат. Ты это делаешь перед такими же учащимися, как и ты сам. У них есть чек-лист, который поможет им выполнить приемку твоей работы качественно.

Роль «Школы 21» заключается в том, чтобы обеспечить через последовательность заданий и оптимальный уровень поддержки такую траекторию обучения, при которой ты освоишь не только hard skills, но и научишься самообучаться.

* Не доверяй слухам и предположениям о том, как должно быть оформлено твое решение. Этот документ является единственным источником, к которому стоит обращаться по большинству вопросов.
* Твое решение будет оцениваться другими учащимися.
* Подлежат оцениванию только те файлы, которые ты сдал на проверку.
* Cдавай на проверку только те файлы, что были указаны в задании.
* Не забывай, что у тебя есть доступ к Интернету и поисковым системам.
* Будь внимателен к примерам, указанным в этом документе — они могут иметь важные детали, которые не были оговорены другим способом.
* И да пребудет с тобой Сила!

## Глава III
### Цели

Этот и следующий проекты могут быть сложными для понимания. Мы непосредственно подошли к предиктивному анализу, в центре которого лежат алгоритмы машинного обучения. «Под капотом» у них заложена серьезная математика, в которую мы вдаваться сильно не будем: чтобы ездить на автомобиле, совсем необязательно знать, как устроен двигатель внутреннего сгорания. При этом, мы заложили в материал моменты, которые помогут тебе интуитивно понять примерный способ работы этих алгоритмов.

## Глава IV
### Задание

В этом проекте очень много новых терминов и понятий, которые необходимо осмыслить и разобрать.

Сегодня ты будешь работать с тремя разными моделями для совершенно разных задач. Ты найдешь ознакомительную информацию для работы с ними в трех ноутбуках. Материалы к этому проекту лежат по [ссылке](https://disk.yandex.ru/d/6yRzb4-YvP0bOQ). 
Однако останется еще часть материала, который придется изучить самостоятельно. Ведь наша цель — это по-прежнему выдерживать правильный баланс между автономией и поддержкой.

#### Text to speech (TTS)

Современные IT-решения уже давно способны преобразовывать устную речь в текст. Но часто бизнес нуждается в противоположном — переводе текстовой записи в аудио. Сегодня ты познакомишься с примером таких моделей — silero models.

Более детальную информацию, а также задание ты найдёшь в ноутбуке `src/day-03-tts.ipynb`.

##### Задание 1

Составь небольшую речь, используя примеры выше, и преобразуй ее в звуковой файл.
1. Выбери подходящий голос для озвучивания.
2. Необходимо, чтобы все ударения были правильно проставлены.
3. Используй как можно больше модификаций голоса (про них более подробно можно прочитать по [ссылке](https://github.com/snakers4/silero-models/wiki/SSML)).
4. Сохрани получившийся аудио-файл.

#### Object Detection

Object detection — технология, связанная с компьютерным зрением (computer vision) и обработкой изображений. Ее работа заключается в обнаружении объектов определенных классов на цифровых изображениях и видео. Причем обнаружение объектов происходит через определение их границ на цифровом изображении или видео. Одной из самых известных моделей обнаружения объектов является семейство моделей YOLO (You Only Look Once).

В блокноте `src/day-03-object-detection.ipynb` ты найдёшь детальную информацию по этой модели.

##### Задание 2

1. В папку `datasets/images/task-2` добавь изображения, на которых ты бы хотел протестировать работу алгоритма. Алгоритм обучен на датасете [COCO](https://cocodataset.org/#explore).
2. Протестируй модель на выбранных изображениях. Проанализируй результаты модели.
3. Если на каких-то изображениях модель продемонстрировала неточные предсказания, то опиши возможную причину.

#### Text Summarization

Ежедневно каждый из нас сталкивается с огромным информационным потоком. Нам часто необходимо изучить множество объемных текстов (статей, документов) в ограниченное время. Поэтому в области машинного обучения естественным образом родилась задача автоматического составления аннотации текста.

Речь о суммаризации — автоматическом создании краткого содержания (заголовка, резюме, аннотации) исходного текста. Сегодня мы поработаем с такой моделью и постараемся разобраться в ее сильных и слабых сторонах.

Более детальную информацию ты найдёшь в блокноте `src/day-03-summarization.ipynb`.

##### Задание 3

Попробуй применить нейронную сеть для других текстов. Проанализируй результаты — посмотри, для каких текстов получилось хорошо, для каких не очень.

#### Открытые модели

Сейчас существует множество платформ для обмена моделями и данных в области искусственного интеллекта и нейронных сетей. Одна из них — это HuggingFace. Чтобы узнать о ней больше, [нажми сюда](https://huggingface.co/spaces). На этой платформе можно найти множество доступных моделей для различных задач, таких как обработка естественного языка, компьютерное зрение, генерация текста и многое другое.

##### Задание 4

Изучи доступные модели на Hugging Face и выбери несколько (3-4), которые тебе интересны. Попробуй их на различных задачах, таких как обработка естественного языка, компьютерное зрение или генерация текста. **Протестировать их можно на самой платформе Hugging Face.**

Оцени результаты и сравни модели между собой. В блокноте `src/day-03-model-review.ipynb` создай отчет о своем опыте использования моделей на Hugging Face. 

В отчете приложи ссылку на модель, опиши, какую задачу решает модель, для решения каких задач её можно применять. Также приложи скриншоты использования модели.

##### Задание 5

Поищи в Интернете различные открытые модели/сервисы с ИИ, которые могут принести **лично тебе** какую-либо пользу. \
*P.S. Чур не ChatGPT.* \
Оставь ссылки на эти ресурсы в блокноте и расскажи, как они тебе помогают.

## Глава V
### Сдача работы и проверка

1. Сохрани решения в файле day-03-object-detection.ipynb, day-03-summarization.ipynb, day-03-tts.ipynb. 
2. Сохрани обзор в файле day-03-model-review.ipynb. 
3. Затем скачай эти файлы из Google Colab. Для этого нажми на кнопку «Файлы» на панели меню --> «Загрузить как» --> Формат .ipynb
3. Загрузи файлы в любое облачное хранилище (например, Яндекс Диск или Google Диск) и предоставь общий доступ «Читателя» по ссылке. Затем скопируй ссылки.
4. Cоздай документ формата .docx или .pdf и вставь в него ссылки на твое решения.
5. Прикрепи документ в раздел «Решение» на платформе.


