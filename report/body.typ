#set text(
  size: 14pt
)
#set math.equation(numbering: "(1)")

#set page(numbering: "1")
#show link: underline


= Введение

Спецификой ритейла является высокая волатильность спроса и сильно ограниченное время реализации товаров [1]. В этой связи прогнозирование спроса является критически важной задачей для бизнеса, для решения которой используются методы статистики, машинного обучения и нейронных сетей [2].  



Прогнозирование временных рядов, в частности задача предсказания спроса, является классической задачей регрессии. Требуется найти отображение, которое данному временному ряду - последовательности $y_1, dots, y_t$ сопоставит набор предсказаний - $overline(y)_(y+1), dots, overline(y)_(t+H)$, минимизировав разницу между предсказаниями и настоящими значениями $y_1, dots, y_(t+H)$\







Задачу предсказания временных рядов решают разными способами. Во-первых, это "модели временных рядов", предполагающие различные виды линейной зависимости между членами временного ряда. Во-вторых, это методы машинного обучения, такие как линейная регрессия, решающие деревья, kernel regression, градиентный бустинг. Наконец, в последние годы для предсказания временных рядов стали активно использоваться нейрости (aritificial neural networks) [1].


Особенностью спроса как временного ряда является то, что фирма может влиять на некоторые из признаков, задающих спрос, а именно на признаки связанные с ценой. В этой связи методы прикладной математики активно применяются для решения задач оптимизации, свзяанных с максимизацией прибыли при известной функции спроса. В честности, модели спроса основанные на эластичности по цене (elasticity based demand function, EDF) [7].


Поскольку задача непосредственно связана с выгодой для бизнеса, открытых исследований на эту тему немного: компании, продающие свои консультационные услуги по выставлению оптимальных цен, не заинтересованы в открытости методов вычисления эластичностей спроса.


В этом исследовании предлагается использование моделей временных рядов SARIMAX для решения задачи предсказания спроса и оптимизации прибыли ритейлера.



== ARIMA

В качестве базовой модели выбрано Auto-regressive integrated moving average, одну из самых популярных статистических моделей для предсказания временных рядов [6]. ARIMA($p, d, q$) предполагает следующую зависимость между членами временного ряда:

$ y_t = alpha_1 y_(y-1) + ... + alpha_p y_(t-p) \
+ beta_1 epsilon_(1) + dots + beta_q epsilon_(t-q)  + epsilon_t $
где $y_t$ - значение в момент $t$, $epsilon_i$ - "белый шум" (компонента ряда, которую невозможно предсказать) в момент $t$








== VAR

Вместе с тем, кажется разумным, что продажи одних товаров могут влиять на продажи других в разной степени. Как минимум, продажи той же категории товаров в предыдущие периоды влияют на продажи этой категории сегодня больше, чем продажи других категорий. Какие-то товары могут являться комплементами для других: спрос на машины приведет к увеличению спроса на бензин.Чтобы учитывать эти взаимосвязи, можно применить модель векторной авторегрессии (VAR), что и было сделано. 



$
y_t = c + A_1y_(t-1) + A_2 y_(t-2) + dots + A_p y_(t-p) + epsilon_t,\
A in R^(k times k),\
y_i, c, epsilon_i in R^\k
$




Данная модель также широко применяется для решения поставленной задачи [2, 3].




== RNN 

Архитектура RNN предназначена для работы с последовательностями и решает более широкий класс задач, в том числе прогнозирование временных рядов. Отличительной особенностью является наличие скрытого состояния - вектора, хранящего информацию о контексте, то есть всех предыдущих элементах последовательности. Однако стандартные RNN сталкиваются с проблемой затухающих градиентов (vanishing gradients), из-за которой ее способность работать с длинными последовательностями сильно ограничена.




== LSTM

Для решения проблемы затухающих градиентов была была предложена архитектура долгосрочной кратковременной памяти (Long Short-Term Memory, LSTM), предложенная Хохрайтером и Шмидхубером в 1997 году. Блоки LSTM хранят свое скрытое состояние отдельно от долгосрочной памяти модели, а механизмы "запоминания" и "забывания" - вентили (gates) - контролировать, что будет храниться в скрытом состоянии. 







= Постановка задачи

== Неформальная постановка задачи





Как уже было сказано, задача регрессии успешно решается различными способами (часть из которых описана выше). Вместе с тем, если рассматривать процесс деятельности ритейлера не со стороны наблюдателя, а со стороны ритейлера, возникает более общая задача - задача максимизации прибыли. Обучая модель регрессии, мы получаем ответ на вопрос: как зависит спрос от различных факторов, которые на него влияют? Среди этих факторов - цена на товар, которую на самом деле фирма устанавливает сама. Поэтому с точки зрения ритейлера можно говорить не только о поиске отображения из пространства признаков в пространство объектов, но и о поиске оптимального значения цены, которое фирме стоит выставить. 

== Формальная постановка задачи


Для начала требуется решить задачу регрессии: по данной выборке $X in RR^(n times m)$ построить отображение $f: RR^n -> RR$, которое приближает функцию спроса $y(x), x in RR^n$, минимизируя функцию потерь. В нашем случае в качестве loss function $cal(L)(X; omega, b)$ выбрана $"MAPE"$ - mean absolute mercentage error. Зависимость спроса от признаков предполагается линейной: 

$
y = omega X + b
$

И задача заключается в 
$
  cal(L)(X; omega, b) -> min omega in RR^n, b in RR
$где 
$
cal(L)(X; omega, b) = 1 / m Sigma |(hat(x_i) - x_i) / x_i|
$

Далее для простоты иногда будем пропускать $b$, потому что свободный член можно без ограничения общности воспринимать как коэффициент для признака, значения которого равно 1 у всех объектов выборки. 



Введем следующие обозначения:
$
p_t - "цена товара в период" t, \
Delta_p = (p_t - p_(t - 1)) / p_(t-1), \
T - "длина сезона для сезонной модели" \
a in RR^n - "все остальные признаки, на которые фирма не может влиять", \
"TR"_t = p_t * y_t, \
"TC"_t - "издержки фирмы", \
pi = "TR" - "TC"
$

Требуется решить задачу оптимизации: 

$
  pi-> max Delta_p
$


== Экономические предпосылки

Чтобы такая задача имела смысл, требуется выполнение некоторых экономических условий, среди которых есть 3 основных:

1) Эластичный спрос: спрос на рассматриваемый товар должен сильно реагировать на изменение цены, чтобы можно было влиять на спрос путем ее изменения. 

2) Волатильность цен: модель предполагает изменение цен каждую неделю.

3) Высокая степень рыночной власти. Как известно из экономической теории, на конкурентном рынке фирма является "ценополучателем" (price-taker), поэтому не может оптимизировать свою прибыль по цене. 






= Предлагаемая модель 

== Описание модели

На текущий момент большинство исследователей, моделирующих спрос с помощью линейных моделей, предполагают используют аппарат линейной регрессии. В этой работе предлагается использовать аппарат интегрированной модели авторегрессии-скользящего среднего (ARIMA) для моделирования спроса.


Базовая персия модели рассматривает спрос $Y_t$  как временной ряд, значение которого зависит от признаков, которые можно разделить на несколько групп:

- лаговые признаки (auto-regressive features)
- ошибки модели в предыдущие периоды (moving-average features)
- сезонность
- остаточный компонент (residuals)
- $Delta_p$ - изменение цены

В виде формулы это можно изобразить в следующем виде: 
$
  y_t = alpha_1 y_(t-1) + dots + alpha_k y_(t - k) \
  + beta_1 epsilon_(t-1) + dots + beta_(t - m) epsilon_(t-m) \
  + gamma_1 y_(y - T) + dots + gamma_l y_(t - l T) \
  + alpha Delta_p + beta epsilon_t
  
$

С использованием обозначений выше:

$
  y_t = alpha Delta_p + beta epsilon_t + a^T x
$ <edf>



Таким  образом, модель можно представить в виде $"SARIMA"(p, d, q, (P, D, Q)_T)$ с экзогенной переменной $Delta_p$. Такую модель также называют $"SARIMAX"$ - сезонная $"ARIMA"$ с вектором $mono(x)$ в качестве экзогенной переменной [8]. 



== Задача оптимизации 

Так как оптимизация производится по $Delta_p$, издержки фирмы являются константой, поэтому достаточно максимизировать величину

$
  "TR" = p_(t - 1) (1 + Delta_p) times y_t (Delta_p)
$



Так как функция спроса предполагается линейной (см. @edf), решение задачи оптимизации выписывается явным образом:

$
  (diff "TR") / (diff Delta_p) = p_(t - 1) y_t (Delta_p) + p_(t-1) (1 + Delta_p) y'_t (Delta_p) = \ = 
  p_(t - 1) y_t (Delta_p) + alpha p_(t-1) (1 + Delta_p)  = 0\
  ==> a^T x + alpha Delta_p + alpha (1 +  Delta_p) = 0 \ 
  ==> Delta_p = (- a^T x - alpha) / (2 alpha)
$


Отдельно требуется рассмотреть случай интегрированной модели, где $alpha$ является коэффициентом при $Delta_p - L^1 Delta_p$, где $L^i$ - лаговый оператор:

$
  y_t = alpha (Delta_p - L^1 Delta_p) + a^T x + beta epsilon_t
$

Задача сводится к предыдущей, так как $- alpha L^1 Delta_p$ являеися константой.


























= Применение модели


== Используемые данные

Для тестирования модели используется датасет о продажах авокадо в США с платформы #link("https://www.kaggle.com/datasets/vakhariapujan/avocado-prices-and-sales-volume-2015-2023")[Kaggle]. Данные о продажах авокадо подходят для нашей задачи, так как они отражают волатильные цены, которые часто меняются еженедельно в зависимости от поставок, спроса и акций ритейлеров, что позволяет моделировать влияние цен на спрос. Доступность еженедельных данных о ценах и объемах через источники вроде #link("https://www.usda.gov")[USDA] и #link("https://hassavocadoboard.com")[Hass Avocado Board] делает их удобными для анализа. Хотя рынок авокадо не является чистой монополией, крупные ритейлеры или брендированные поставщики обладают достаточной рыночной властью, чтобы экспериментировать с ценами и оптимизировать прибыль, что соответствует целям исследования.




== Подбор гиперпараметров


Для выбора гиперпараметров модели построены графики автокорреляционной и частичной автокорреляционной функций: как известно, оптимальным значением p в модели AR(p) считается последний значимый пик ACF, а оптимальным значением q в MA(q) - последний значимый пик PACF [5]. 



Затем процесс подбора гиперпараметров автоматизирован с помощью библиотеки sktime. 
Для подбора оптимальных значений p, d, q использовался grid search - перебор всех разумных комбинаций гиперпараметров - с кросс-валидацией типа: модель последовательно обучалась на данных за предыдущие 2 года и предсказывала на 1 неделю вперед.   Для наглядности приведен пример кросс-валидации временного ряда, в котором размер тренировочной выборки на каждом шаге равен 5, а размер тестовой 3. Из-за того, что данных о недельных продажах не может быть много, такой подход особенно ценен в этой задаче, так как позволяет использовать одни и те же данные сначала в обучающей выборке, затем в тестовой. 

$#figure(
  image(
  "cross-validation.png"
),
  caption: [\* - тренировочные данные, х - тестовые данные]
),$


$#figure(
  image(
  "../src/cv_results.png"
),
  caption: [Результаты кросс-валидации]
),$







== Интерпретация результатов



По итогам проведенных экспериментов наилучшие предсказания получились у модели $"ARIMAX(3, 1, 4)"$. 




$#figure(
  image(
  "../src/graphics/arima314.png",
  height: 30%,
  // fit: "stretch"
),
  caption: [результаты ARIMAX(p=3, d=1, q=4, X=$Delta_p$)]
),$


\
$#figure(
  image(
  "../src/graphics/residual_component_arimax_314.png",
  height: 45%
),
  caption: [Ошибки модели на тестовой выборке]
),$

$#figure(
  image(
  "../src/graphics/arima314_resid_autocorrelation.png",
  height: 45%
),
  caption: [Автокорреляционная функция ошибок модели]
),$



Ключевое "правильное" свойство ошибок - некоррелированность - выполняется: синей областью на графике автокорреляционной функции ряда ошибок модели отмечен интервал, в который попадают незначимые корреляции (автокоррелированность ошибок говорит о том, что модель не учитывает часть взаимосвязей между элементами временного ряда).


== Мудрость толпы

Оценка "советов" модели об установке оптимальной цены - более сложная задача - чтобы оценить ее качество, нужно было бы провести $A "/" B $ тест, а такой возможности у исследователей нет. 


Поэтому будем считать, что продавцы авокадо выставляют правильные или почти правильные цены на свой продукт согласно принципу "мудрости толпы" [9]: среднее большого количества независимых предсказаний должно совпадать с истинным значением. В нашем случае среднее предсказаний оптимальной цены фирмами-участниками рынка должно быть похоже на действительно оптимальное значение. 



$#figure(
  image(
  "../src/graphics/optimal_pricing.png",
),
  caption: [Средняя цена на рынке и оптимальная цена согласно предсказанию модели]
),$



В целом модель предлагает цены, похожие на настоящие - это хорошее свойство. С другой стороны, модель предлагает более активную ценовую политику, чем в среднем по рынку - это закономерно, так как не учитывается конкуренция на рынке. 





#align(center)[$#figure(
  table(
    columns: 2,
    [], [std],
    [Actual price], [0.0628],
    [Optimal price], [0.1555],
  ),
  caption: "Стандартное отклонение реальных и оптимальных цен"
)
$]

Ниже представлен график, показывающий, какое улучшение прибыли "обещает" модель исходя из своих представлений об эластичности спроса:


$#figure(
  image(
  "../src/graphics/optimal_and_real.png",
),
  caption: [Предсказанная выручка при оптимальной цене и реальной цене]
),$

\

На графике выше по вертикальной оси - выручка при оптимальной цене и предсказанном для этой цены спросе, по горизонтальной - соответственно, недели. 







== Моделирование спроса с LSTM






Далее в качестве альтернативного подхода для сравнения используется простейшая нейросеть: 2 LSTM слоя с $200$ и $30$ нейронами соответственно и Dropout регуляризация, деактивирующая $20%$ нейронов на каждом слое (коэффициент выброса $0.2$). Модель была обучена в течение $50$ эпох.


$#figure(
  image(
  "../src/graphics/lstm_forecasts.png",
),
  caption: [Средняя цена на рынке и оптимальная цена согласно предсказанию модели]
),$


\ \


#align(center)[$#figure(
  table(
    columns: 2,
    [Model], [test MAPE score],
    [Naive], [0.1183],
    [SARIMAX], [0.0628],
    [LSTM], [0.0381],
  ),
  caption: "Сравнение результатов двух моделей"
)
$]

Как видим, даже очень простая по современным меркам нейросеть смогла превзойти классическую модель. Причем, безусловно, этот результат тоже можно улучшить. Обе модели превосходят бенчмарк-наивный предсказатель (всегда предсказывающий $y_t = y_(t - 1)$)



== Выводы

Полученная модель дает качественные предсказания, хорошо решает задачу предсказания спроса. Решение оптимизационной задачи похоже на настоящие цены и хорошо бы подошло для фирмы-монополиста, которая может проводить активную ценовую политику. 




#pagebreak()
= Источники

== Список литературы



[1] Da Veiga CP, Da Veiga CR, Catapan A, Tortato U, Da Silva WV. Demand forecasting in food retail: A comparison between the Holt-Winters and ARIMA models. WSEAS transactions on business and economics. 2014 Jan;11(1):608-14.

[2] Tsolacos S. Econometric modelling and forecasting of new retail development. Journal of Property Research. 1998 Jan 1;15(4):265-83.

[3] Brooks C, Tsolacos S. Forecasting models of retail rents. Environment and Planning A. 2000 Oct;32(10):1825-39.

[4] Hyndman RJ. Forecasting: principles and practice. OTexts; 2018.
URL: https://otexts.com/fpp2/accuracy.html

[5] Robert Nau, Identifying the numbers of AR or MA terms in an ARIMA model, Duke University, 2020
URL: https://people.duke.edu/~rnau/411arim3.htm

[6] Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: Forecasting and control (5th ed). Hoboken, New Jersey: John Wiley & Sons.

[7]Elasticity Based Demand Forecasting and Price Optimization for Online Retail, Chengcheng Liu, M ́aty ́as A. Sustik Walmart Labs, San Bruno, CA, June 17, 2021.
URL: https://arxiv.org/pdf/2106.08274

[8] Vagropoulos SI, Chouliaras GI, Kardakos EG, Simoglou CK, Bakirtzis AG. Comparison of SARIMAX, SARIMA, modified SARIMA and ANN-based models for short-term PV generation forecasting. In2016 IEEE international energy conference (ENERGYCON) 2016 Apr 4 (pp. 1-6). IEEE.

[9] Wagner C, Vinaimont T. Evaluating the wisdom of crowds. Proceedings of Issues in Information Systems. 2010 Sep;11(1):724-32.




== Приложения

[1] Репозиторий проекта\ $space$ URL: https://github.com/chagrygoris/retail_forecasts











