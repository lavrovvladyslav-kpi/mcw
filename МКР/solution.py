"""МКР з Python for Data Science — наскрізний кейс «Метеослужба».

ШАБЛОН ДЛЯ СТУДЕНТА. Заповніть кожен пункт у блоках 1–4 та допишіть
ВИСНОВКИ у docstring наприкінці файлу.

Перед запуском скрипта підніміть СВІЙ Docker-контейнер з MySQL:

    docker pull <DOCKER_USER>/pfds-mkr-g<N>-<NN>
    docker run -d -p 3306:3306 --name mkr <DOCKER_USER>/pfds-mkr-g<N>-<NN>

(g<N>-<NN> — ваші група і номер у журналі, видається викладачем)

Потім чекайте ~30 секунд на ініціалізацію MySQL і запускайте:

    python solution.py

Графіки зберігаються в підпапку `plots/` поряд зі скриптом.
"""

# ====================================================================
# Прізвище, ім'я, по батькові: Лавров Владислав Володимрович_______
# Група:                       КІ-32_______________________________
# Дата виконання:              14.05.2026__________________________
# ====================================================================

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "127.0.0.1"
DB_PORT = 33306
DB_NAME = "meteo"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 12, delay: float = 2.5) -> pd.DataFrame:
    """Підключитися до MySQL і завантажити таблицю observations.

    MySQL-контейнер на старті виконує LOAD DATA INFILE, що займає
    ~20–30 секунд. Тому робимо retry-цикл — перші спроби очікувано
    падають з OperationalError (server not ready).
    """
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(url)
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Підключено до MySQL з {attempt}-ї спроби. Рядків: {len(df)}")
            return df
        except OperationalError:
            if attempt == retries:
                raise
            print(f"  MySQL ще не готова (спроба {attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


# ====================================================================
# БЛОК 1. NumPy (15 балів)
# ====================================================================
# Працюємо з СИРИМИ даними (до очищення в Pandas). Використовуємо
# тільки numpy-арифметику, без pandas-арифметики.

def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy")

    # необіхідні колонки з датафрейму
    obs_id = df_raw['obs_id'].to_numpy(dtype=int)
    datetime = df_raw['datetime'].to_numpy()
    temperature_c = df_raw['temperature_c'].to_numpy(dtype=float)
    humidity_pct = df_raw['humidity_pct'].to_numpy(dtype=float)
    wind_speed_ms = df_raw['wind_speed_ms'].to_numpy(dtype=float)

    # 1) Побудувати np.array apparent temperature за формулою:
    #    T_app = T - (100 - RH) / 5
    #    Працюйте з temperature_c і humidity_pct як з np.array.
    # TODO:
    apparent = temperature_c - (100 - humidity_pct) / 5
    print(f"1) T_app: len={len(apparent)}, min={np.nanmin(apparent):.2f}, max={np.nanmax(apparent):.2f}")

    # 2) Замінити викидні значення:
    #    - temperature_c > 60 або < -60   -> np.nan
    #    - wind_speed_ms > 100            -> np.nan
    #    Використати np.where.
    # TODO:
    temp_count = np.sum((temperature_c > 60) | (temperature_c < -60))
    wind_count = np.sum(wind_speed_ms > 100)

    temperature_clean = np.where((temperature_c > 60) | (temperature_c < -60), np.nan, temperature_c)
    wind_clean = np.where(wind_speed_ms > 100, np.nan, wind_speed_ms)

    print(f"2) Викидів температури замінено: {temp_count}")
    print(f"   Викидів вітру замінено:       {wind_count}")

    # 3) Порахувати mean / median / std температури ВРУЧНУ
    #    (без pandas .describe(), ігноруючи NaN). Дозволені np.nansum,
    #    np.nanmedian, np.sqrt, маски тощо.
    # TODO:
    mask = np.isnan(temperature_clean)
    temperature_clean_filtered = temperature_clean[~mask]
    length = len(temperature_clean_filtered)

    mean_t = np.nansum(temperature_clean) / length
    median_t = np.nanmedian(temperature_clean)
    std_t = np.sqrt(np.nansum((temperature_clean - mean_t) ** 2) / length)
    print(f"3) mean={mean_t:.3f}  median={median_t:.3f}  std={std_t:.3f}")

    # 4) Маска: скільки спостережень "морозних" (T<0) і "жарких" (T>30).
    # TODO:
    mask_frost = temperature_clean < 0
    mask_hot = temperature_clean > 30

    n_frost = len(temperature_clean[mask_frost])
    n_hot = len(temperature_clean[mask_hot])
    print(f"4) морозних: {n_frost}    жарких: {n_hot}")

    # 5) argmax / argmin температури -> повернути obs_id і datetime
    #    цих рядків. Підказка: np.nanargmax / np.nanargmin.
    # TODO:
    max_temp_c_id = np.nanargmax(temperature_clean)
    min_temp_c_id = np.nanargmin(temperature_clean)

    print(f"5) Максимум температури: {temperature_clean[max_temp_c_id]} , obs_id={obs_id[max_temp_c_id]}, datetime={datetime[max_temp_c_id]}")
    print(f"   Мінімум температури:  {temperature_clean[min_temp_c_id]}, obs_id={obs_id[min_temp_c_id]}, datetime={datetime[min_temp_c_id]}")


# ====================================================================
# БЛОК 2. Pandas — очищення (20 балів)
# ====================================================================

def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — очищення")

    rows_before = len(df_raw)
    df = df_raw.copy()

    # 1) Перевірте типи (info), статистику (describe).
    # TODO:
    print("1) Перевірка типів:")
    df.info()
    print("   Перевірка статистики:")
    print(df.describe())

    # 2) Перевести datetime у тип datetime та зробити індексом.
    # TODO:
    df['datetime'] = pd.to_datetime(df['datetime']) # переведення тип datetime
    df = df.set_index('datetime') # зробив індексом

    # 3) Видалити повні дублі рядків.
    # TODO:
    length_1 = len(df) # до видалення
    df = df.drop_duplicates(keep='first')
    n_dups = length_1 - len(df)
    print(f"2) drop_duplicates: видалено {n_dups}")

    # 4) Заповнити NaN у humidity_pct МЕДІАНОЮ ПО МІСЯЦЮ В МЕЖАХ МІСТА.
    #    Підказка: groupby([city, month]).transform('median'),
    #    де month = df.index.month.
    # TODO:
    nan_count = df['humidity_pct'].isna().sum()
    median_2 = df.groupby(['city', df.index.month])['humidity_pct'].transform('median')
    df = df.fillna(value = {'humidity_pct': median_2}) # заповнення медіаною
    n_filled = nan_count - df['humidity_pct'].isna().sum()
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    # 5) Прибрати фізичні викиди:
    #    - temperature_c має бути в [-60, 60]
    #    - wind_speed_ms (де не NaN) має бути в [0, 60]
    # TODO:
    length_1 = len(df) # до видалення
    temp_correct = (df['temperature_c'] >= -60) & (df['temperature_c'] <= 60)
    wsp_correct = df['wind_speed_ms'].isna() | ((df['wind_speed_ms'] >= 0) & (df['wind_speed_ms'] <= 60))
    df = df[temp_correct & wsp_correct]
    n_outliers = length_1 - len(df)
    print(f"4) Видалено фізичних викидів: {n_outliers}")

    # 6) Звіт очищення.
    print(f"\n   Звіт: {rows_before} → {len(df)} рядків")

    return df


# ====================================================================
# БЛОК 3. Pandas — аналітика (30 балів)
# ====================================================================

def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — аналітика")

    # 1) Середня температура по містах (sort_values).
    #    Хто найтепліше / найхолодніше?
    # TODO:
    by_city_temp = df.groupby('city')['temperature_c'].mean().sort_values(ascending=False) # додав ascending=False, щоб виводило найтепліше -> найхолодніше
    print("1) Середня T по містах:")
    print(by_city_temp.round(2).to_string())

    # 2) Сумарні опади по містах. Хто найвологіше?
    # TODO:
    by_city_precip = df.groupby('city')['precipitation_mm'].sum().sort_values(ascending=False) # ascending=False, щоб найвологіше місто було першим
    print("\n2) Сумарні опади по містах:")
    print(by_city_precip.round(1).to_string())

    # 3) Місячна середня температура: resample('ME').mean()
    #    (для старих pandas — 'M' замість 'ME').
    # TODO:
    monthly_mean = df['temperature_c'].resample('ME').mean()
    print(f"\n3) Місячна середня T ({len(monthly_mean)} точок):")
    print(monthly_mean.round(2).to_string())

    # 4) Pivot: місто × місяць, значення = середня T.
    # TODO:
    pivot = df.pivot_table(values = 'temperature_c', index = 'city', columns = df.index.month) # aggfunc='mean' за замовченням
    print("\n4) Pivot місто × місяць:")
    print(pivot.round(1).to_string())

    # 5) Кількість днів з опадами > 5 мм по містах.
    #    Підказка: спочатку зробіть денні суми по місту, потім порахуйте.
    # TODO:
    date_sum = df.groupby(['city', df.index.date])['precipitation_mm'].sum() # денні суми по кожному місту
    days5_count = date_sum[date_sum > 5] # умова > 5 мм
    rainy_days = days5_count.groupby('city').count() # кількість днів по містах
    print("\n5) Дні з опадами > 5 мм:")
    print(rainy_days.to_string())

    # 6) Знайти аномальний місяць.
    #    Підхід: для кожного календарного місяця (1..12) обчислити
    #    "норму" як середню по тому ж місяцю обох років, потім знайти
    #    (year, month) з максимальним |відхиленням| від норми.
    # TODO:
    norm = df.groupby(df.index.month)['temperature_c'].mean() # норма для кожного календарного місяця
    fact = df.groupby([df.index.year, df.index.month])['temperature_c'].mean() # факт темп. для кожного календарного місяця
    deviation = fact.sub(norm, level = 1) # відхилення 
    anomaly_month = deviation.abs().idxmax() # аномальний місяць 
    anomaly_dev = deviation[anomaly_month] # відхилення в аномальний місяць
    print(f"\n6) Аномальний місяць: {anomaly_month[0]}-{anomaly_month[1]:02d}  відхилення = {anomaly_dev:+.2f}°C") # тріщки змінив вивід anomaly_month, бо виглядив некоректно

    # вирішив додати бонусний блок для відповіді на питання:
    # Який кліматичний регіон стабільніший за температурою (за std)?
    region_std = df.groupby('city')['temperature_c'].std().sort_values()
    print("\nСтабільність клімату в регіонах (за std):")
    print(region_std.round(3).to_string())

    return {
        "by_city_temp": by_city_temp,
        "by_city_precip": by_city_precip,
        "monthly_mean": monthly_mean,
        "pivot": pivot,
    }


# ====================================================================
# БЛОК 4. Matplotlib + інтерпретація (35 балів)
# ====================================================================

def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Matplotlib")

    # корисні дані з блоку 3
    pivot = analytics['pivot'] # для Графік 1 та Графік 4
    by_city_precip = analytics['by_city_precip'] # для Графік 2

    #

    # Графік 1: line — місячна динаміка температури по 3 обраних містах.
    # Вимоги: title, xlabel, ylabel, legend, форматування дат.
    # TODO:
    fig, ax = plt.subplots(figsize=(11, 5))
    
    cities = pivot.index[:3] # вирішив взяти просто перші три міста
    for city in cities:
        ax.plot(pivot.columns, pivot.loc[city], marker='o', label=city) # pivot.loc[city] - ось y

    ax.set_title("Місячна динаміка температури")
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Температура, °C")
    ax.legend()

    # форматування дат
    ax.set_xticks(range(1, 13))
    # назви місяців
    month_names = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер', 
                   'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']
    ax.set_xticklabels(month_names)

    ax.grid()

    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    #

    # Графік 2: bar — сумарні опади по містах.
    # TODO:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    by_city_precip.plot(kind='bar', color='yellow', edgecolor='black', zorder = 2) # вказав zorder, бо не подобалось, що розмітка була поверх стовпчиків

    ax.set_title("Сумарні опади по містах")
    ax.set_xlabel("Місто")
    ax.set_ylabel("Опади, мм")

    ax.grid(axis='y', zorder=0)

    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    #

    # Графік 3: hist — розподіл температур з вертикальними лініями
    #    mean і median.
    # TODO:
    fig, ax = plt.subplots(figsize=(9, 5))
    
    df['temperature_c'].plot(kind='hist', bins=15, color='green', edgecolor='black', label = 'Розподіл') # вказав bins, бо за замовчуванням (на мій погляд) мало поділів
    
    mean_value = df['temperature_c'].mean()
    median_value = df['temperature_c'].median()
    
    ax.axvline(mean_value, color='red', label=f'Mean: {mean_value:.1f}°C')
    ax.axvline(median_value, color='blue', label=f'Median: {median_value:.1f}°C')
    
    ax.set_title("Гістограма розподілу температур")
    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Кількість спостережень")
    ax.legend()

    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    #

    # Графік 4: heatmap pivot місто × місяць (plt.imshow).
    #    Не забудьте colorbar і підписи осей.
    # TODO:
    fig, ax = plt.subplots(figsize=(11, 5))
    
    hmap = ax.imshow(pivot.values, cmap='coolwarm') # матриця кольорів
    fig.colorbar(hmap, ax=ax, label="Температура, °C")

    ax.set_title("Теплова карта")

    ax.set_xlabel("Місяць")
    ax.set_xticks(range(len(pivot.columns)))
    month_names = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер', 
                   'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']
    ax.set_xticklabels(month_names)

    ax.set_ylabel("Місто")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    #

    print(f"4 графіки збережені в {PLOTS_DIR}/")



# ====================================================================

def main() -> None:
    df_raw = load_observations()
    print(f"Завантажено: shape={df_raw.shape}")

    block_1_numpy(df_raw)
    df_clean = block_2_cleaning(df_raw)
    analytics = block_3_analytics(df_clean)
    block_4_plots(df_clean, analytics)


if __name__ == "__main__":
    main()


"""
ВИСНОВКИ (5–8 речень).

Напишіть тут вашу інтерпретацію даних. Орієнтири:
- Яке місто найтепліше/найхолодніше? Як ви це поясните географічно?
- Як виражена сезонність температури?
- Який місяць аномальний? Це хвиля спеки чи холоду? Як ви це визначили?
- Який кліматичний регіон стабільніший за температурою (за std)?
- 1–2 рекомендації: що б ви порадили на основі цих даних
  (наприклад, де варто будувати склади-холодильники, яку статтю
  витрат компанії важливо враховувати взимку тощо).

Ваш текст:
Усі результати взяті з результату виконання в терміналі, а також графіків.
Найтеплішим містом виявився Київ (середня температура 11.91°C), що географічно 
можна пояснити сильним кліматичним ефектом щільної забудови мегаполіса. Натомість найхолоднішою 
в середньому стала Одеса (7.44°C), оскільки морський клімат та постійні бризи з Чорного моря 
згладжують літні температурні максимуми. Щодо сезонності температури, то мінімуми явно фіксуються зимою 
у грудні-січні, а пік спеки припадає на період з травня по липень. Аномальним місяцем визначено 
вересень 2024 року із відхиленням +4.24°C. Оскільки значення додатне, це хвиля осінньої спеки, яку було виявлено шляхом 
віднімання багаторічної норми від фактичної температури цього місяця. 
На основі даних бізнесу можна дати дві рекомендації. 
По-перше, при будівництві складів у Харкові (понад 737.4 мм опадів та 45 днів злив) критично важливо інвестувати 
у системи водовідведення та гідроізоляції. По-друге, в Одеській області склади-холодильники потребуватимуть 
менших витрат електроенергії на охолодження влітку, проте місцевому агросектору необхідно закладати великий бюджет 
на штучне зрошення через невеликі 205 мм опадів.
"""

# asterindex/pfds-mkr-g5-11:latest