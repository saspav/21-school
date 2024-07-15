# День 02 — Введение в машинное обучение
## Машинное обучение

По итогам этого проекта ты сможешь построить модель машинного обучения для задачи классификации, предварительно обработав данные.

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

Наверное, тебе встречалось довольно много разных терминов из сферы анализа данных: искусственный интеллект, машинное обучение, нейронные сети, глубокое обучение. Но чем отличаются друг от друга все эти термины и отличаются ли? 

На самом деле, каждый последующий термин из этого списка является подмножеством предыдущего. 

То есть самый широкий термин из всех перечисленных — искусственный интеллект. Он включает в себя любые техники и алгоритмы, которые способны имитировать человеческое поведение. Это могут быть алгоритмы машинного обучения, а могут быть просто правила, написанные на любом языке программирования в духе «if-then-else». 

Например, еще в 1966 году был создан виртуальный собеседник Элиза, которая имитировала диалог с психотерапевтом. В большинстве случаев она просто перефразировала то, что говорил человек. В некоторых случаях она находила ключевые слова, к которым были привязаны специальные реплики. Несмотря на то, что в программе не использовалось никаких нейронных сетей или алгоритмов машинного обучения, ее можно считать ранним вариантом искусственного интеллекта. 

Та же современная Алиса от Яндекса отчасти построена на тех же принципах и правилах, хотя и использует уже алгоритмы машинного обучения.

![pic-1](misc/images/pic-1.png)

Что же такое машинное обучение? Машинное обучение включает в себя статистические алгоритмы, которые автоматизируют процесс создания этих самых правил: их больше не надо прописывать вручную. 

Например, обученная модель способна сама распознавать эмоциональное состояние человека по реплике. Реплики могут быть разными, содержать множество разных ключевых слов, но модель способна почти во всех из них правильно определить эмоциональную окраску и соответствующим образом среагировать.

Подмножеством алгоритмов машинного обучения являются нейронные сети. Создатели этих алгоритмов вдохновлялись тем, как устроен человеческий мозг (тем не менее, нейронные сети достаточно далеки от полного подобия).

А подмножеством нейронных сетей являются алгоритмы deep learning. Это тоже нейронные сети, но обладающие большим количеством слоев (большим количеством уровней иерархии). По этой причине они называются «глубокими».

![pic-2](misc/images/pic-2.png)


<details><summary>Расписание проектов</summary>

**День 00. Сбор данных** \
Данные находятся в разрозненных источниках. Их надо собрать и объединить в единый датасет и разобраться, какие данные у нас есть.

**День 01. Анализ данных:** \
Работа с простыми дескриптивными статистиками. Цель — чуть лучше понять анализ данных, выявить дополнительные проблемы с качеством данных и решить их. Ты построишь гистограммы, разные графики, чтобы еще лучше понять, как устроены данные.
Всё это может дать идеи для создания новых продуктов.

**День 02. Машинное обучение ← Ты находишься здесь:** \
Займемся задачей предсказания оттока клиентов. Подготовим данные к обучению.
Обучим модель машинного обучения. Оценим качество предсказания.

**День 03. Глубокое обучение:** \
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
### Цель

Этот и следующий проекты могут быть сложными для понимания. Мы непосредственно подошли к предиктивному анализу, в центре которого лежат алгоритмы машинного обучения. «Под капотом» у них заложена серьезная математика, в которую мы сильно вдаваться не будем: чтобы ездить на автомобиле, совсем необязательно знать, как устроен двигатель внутреннего сгорания.
 
При этом, мы заложили в материал моменты, которые помогут тебе интуитивно понять примерный способ работы этих алгоритмов.

## Глава IV
### Задание

Машинное обучение можно разделить на две части: с учителем и без учителя.

**Обучение без учителя** — это раздел машинного обучения, в котором модель обучается на неразмеченных данных. В этом случае модель сама ищет закономерности и паттерны в данных без предоставления конкретных ответов или меток. Существует несколько методов обучения без учителя, включая кластеризацию, понижение размерности, поиск аномалий и многие другие. 

Вот некоторые из наиболее распространенных методов обучения без учителя:

* **Кластеризация** — метод для группирования данных в кластеры на основе их сходства.
* **Понижение размерности** — метод, позволяющий уменьшить количество признаков в данных, сохраняя при этом наиболее важные характеристики. 
* **Поиск аномалий** — метод для выявления аномальных данных из генеральной совокупности.

Обучение без учителя широко применяется в таких областях, как анализ данных, кластеризация пользователей, поиск аномалий, сегментация рынка и другие. Оно позволяет извлекать ценную информацию из данных, даже если нет четких ответов или меток для обучения модели.

**Обучение с учителем** — это другой основной подход в машинном обучении, при котором модель обучается на размеченных данных, где каждый пример имеет соответствующий выходной ответ или метки. В обучении с учителем модель учится на основе предоставленных пар «вход-выход» и стремится к минимизации ошибки между предсказанным и истинным значениями.

Задачи обучения с учителем разделяются на:

* **Классификацию** — метод для прогнозирования категориальных выходных значений. 
Например, модель определяет, является ли электронное письмо спамом или не спамом.
* **Регрессию** — метод для прогнозирования непрерывных выходных значений. Например, модель прдесказывает цены недвижимости на основе характеристик дома.

Обучение с учителем широко применяется в различных областях, таких как финансы, медицина, маркетинг, обработка естественного языка и другие. Оно позволяет создавать модели, способные делать точные прогнозы и принимать решения на основе имеющихся данных и меток.

![ml](misc/images/ml.jpg)

Ну, достаточно теории, давай перейдем к заданию!

Ноутбук проекта src/day-02-assignment.ipynb, а также остальные материалы к этому проекту ждут тебя по [ссылке](https://disk.yandex.ru/d/Oy_8duyvXDCqEg). Напомним, что ты продолжаешь работать в Google Colab. MS Excel по-прежнему под запретом.

Перед нами стоит задача классификации. Нам нужно научиться оценивать вероятность того, что пользователь перестанет пользоваться нашими услугами. У тебя есть данные о клиентах, и тебе известны пользователи, которые ранее прекращали пользоваться нашими услугами. Машина, глядя на эти данные, должна найти закономерности и создать модель, предсказывающую цену нового объекта по имеющимся данным. Помимо этого тебе нужно будет измерить качество моделей. 

Для этого разобьем датасет на две части: обучающую выборку (train) и тестовую (test). На обучающей выборке алгоритм будет учиться, а на тестовой мы проверим его реальную точность — на тех примерах, которые алгоритм еще не видел. Ведь суть построения модели заключается в том, чтобы она дальше встроилась в один из реальных процессов, где ранее решение о незнакомых данных принималось каким-то другим алгоритмом (например, оценка человека, работающего с данными), не имея априорной информации о целевом значении. 

Но перед этим нам придется еще поработать с предобработкой данных. Алгоритмы машинного обучения в этом смысле бывают достаточно привередливыми.

## Задание 1
Удали признаки с приставками `charg_inst`, `charg_sale`, `hgid`, `hflat`, `hlid`, а также данные о координатах `latitude_1m` и `longitude_1m`. В удалении признаков тебе поможет метод 
[.drop](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html).

Выведи размерность получившейся таблицы.

## Задание 2
Теперь разберемся, что в каждой из задач для нас будет являться признаком (Х), а что предсказываемой переменной (Y).

Раздели датасет `train` на признаки и предсказываемую величину. Признаки сохрани в переменную `X`, а предсказываемую величину — в переменную `Y`. 

Выведи размерность `X` и `Y`.

## Задание 3
Теперь нам потребуется разделить наши признаки на численные и категориальные. В этом нам поможет функция метод [.select_dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html). 

Численные переменные имеют тип `'number'`, а категориальные `'object'`. Численные переменные сохрани в переменную X_num, а категориальные — в переменную X_cat.

Выведи размерность этих таблиц.

## Задание 4
Некоторые алгоритмы машинного обучения чувствительны к пропускам данных, поэтому нам придется чем-то заполнить отсутствующие данные. Воспользуйся методом [.fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) для того, чтобы заполнить пропуски в численных признаках значением 0.

Затем выполни этот код `X_num.isna().sum().sum()`.


## Задание 5
Примени преобразования **One-Hot Encoding** для категориальных признаков. В этом тебе поможет функция [pd.get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)

Выведи размерность получившейся таблицы.

## Задание 6
Теперь уже преобразованные части требуется снова объединить в единый датасет. Соедини переменные `X_num` и `X_cat` в переменную `X`. Воспользуйся методом [.merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html). Объединение проведи по индексам таблицы.

Выведи размерность получившейся таблицы.

## Задание 7
Раздели итоговый датасет на train и test части. В этом поможет функция [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Разделение проведи с параметрами `test_size=0.2`, `random_state=21`.

Выведи размерности переменных `X_train` и `X_test`.

## Задание 8
Рассчитай точность обученной модели. Для этого с помощью метода `.predict(X_test)` сделай предсказание на тестовой выборке `X_test`. Затем передай предсказанные классы и истинные значения оттока в функцию `accuracy_score`. 

Не подозрительный ли получился результат? Рассчитай долю клиентов, которые **не уйдут**.

## Задание 9
Рассчитай метрику ROC-AUC. Для этого с помощью метода `.predict_proba(X_test)[:,1]` сделай предсказание на тестовой выборке `X_test`. Затем передай вероятности класса и истинные значения оттока в функцию `roc_auc_score`. 

## Глава V
### Сдача работы и проверка

1. Сохрани решения в файле day-02-assignment.ipynb. Затем скачай его из Google Colab. Для этого нажми на кнопку «Файлы» на панели меню --> «Загрузить как» --> Формат .ipynb.
2. Загрузи файл в любое облачное хранилище (например, Яндекс Диск или Google Диск) и предоставь общий доступ «Читателя» по ссылке. Затем скопируй ссылку.
3. Cоздай документ формата .docx или .pdf и вставь в него ссылку на свое решение.
4. Прикрепи документ в раздел «Решение» на платформе.