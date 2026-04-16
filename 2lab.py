import wave
import numpy as np
import matplotlib.pyplot as plt

print("Визуализация и спектральный анализ речевых сигналов")
print()

# Ввод имени файла и открытие — повторяем пока файл не найден
while True:
    try:
        filename = input("Введите имя wav-файла: ")
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")
        exit()
    try:
        wav_file = wave.open(filename, "rb")
        break  # файл открыт успешно — выходим из цикла
    except FileNotFoundError:
        print("  Ошибка! Файл не найден. Попробуйте снова.")
    except wave.Error as e:
        print("  Ошибка! Некорректный WAV-файл:", e, ". Попробуйте снова.")

# Чтение параметров и данных из файла
n_channels = wav_file.getnchannels()  # 1 = моно, 2 = стерео
framerate = wav_file.getframerate()   # частота дискретизации, Гц
sampwidth = wav_file.getsampwidth()   # разрядность: 1=8бит, 2=16бит, 4=32бит
N = wav_file.getnframes()             # количество отсчётов в файле
frames = wav_file.readframes(-1)      # читаем все байты из файла

# Проверка на пустой файл
if N == 0:
    print("Ошибка! WAV-файл не содержит аудиоданных.")
    exit()

# Определяем тип данных по разрядности файла
dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
dtype = dtype_map.get(sampwidth)
if dtype is None:
    print("Ошибка! Неподдерживаемая разрядность файла:", sampwidth * 8, "бит.")
    exit()

# Перевод байтов в числа
signal_full = np.frombuffer(frames, dtype=dtype).astype(np.int32)

# Стерео или моно — берём левый канал
if n_channels == 2:
    signal_full = signal_full[::2]
    print("Файл стерео — используется левый канал.")

# Информация в консоль
print("Частота дискретизации:", framerate, "Гц")
print("Разрядность:", sampwidth * 8, "бит")
print("Всего отсчётов в файле:", len(signal_full))
print("Длительность:", round(len(signal_full) / framerate, 2), "с")
print()

# Ввод количества отсчётов для графиков 1, 2, 4
while True:
    try:
        n = int(input(
            "Введите количество отсчётов"
            " (от 2 до " + str(len(signal_full)) + "): "
        ))
        if 2 <= n <= len(signal_full):
            break
        print("  Введите число от 2 до", len(signal_full), ".")
    except ValueError:
        print("  Ошибка! Введите целое число.")
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")
        exit()

print("Будет отображено", n, "отсчётов на графиках 1, 2, 4.")
print("Спектр строится по всем", len(signal_full), "отсчётам файла.")
print()

# Обрезаем сигнал до n отсчётов — только для визуализации формы сигнала
signal = signal_full[:n]

# Дискретное время в секундах
T = 1 / framerate
time = np.linspace(0.0, n * T, n, endpoint=False)


# График 1. Круговая диаграмма
# Делим диапазон амплитуд на 3 зоны: низкие, средние, высокие

min_val = np.min(signal)
max_val = np.max(signal)
step = (max_val - min_val) / 3

low = np.sum(signal < min_val + step)
mid = np.sum((signal >= min_val + step) & (signal < min_val + 2 * step))
high = np.sum(signal >= min_val + 2 * step)

# Убираем нулевые секторы для корректного отображения
filtered = [
    (v, l)
    for v, l in zip([low, mid, high], ["Низкие", "Средние", "Высокие"])
    if v > 0
]

plt.figure()
if filtered:
    values, labels = zip(*filtered)
    plt.pie(values, labels=labels, autopct="%1.1f%%")
else:
    plt.text(0.5, 0.5, "Нет данных", ha="center", va="center")
plt.title(
    "Круговая диаграмма распределения амплитуд\n"
    "(первые " + str(n) + " отсчётов)"
)


# График 2. Осциллограмма речевого сигнала

plt.figure()
plt.plot(time, signal, color="blue", linestyle="-", linewidth=1)
plt.title("Осциллограмма речевого сигнала")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, у.е.")
plt.grid()


# График 3. Спектр: ln(Re**2 + Jm**2)
# Спектральный анализ выполняется по ВСЕМ отсчётам файла.
# +1e-10 добавляется, чтобы избежать логарифма нуля.

spectrum = np.fft.rfft(signal_full)
re = np.real(spectrum)
im = np.imag(spectrum)
log_spectrum = np.log(re ** 2 + im ** 2 + 1e-10)
freqs = np.fft.rfftfreq(len(signal_full), T)

plt.figure()
plt.plot(freqs, log_spectrum, color="green", linestyle="-", linewidth=1)
plt.title("Спектр: логарифм квадрата модуля ДПФ  ln(Re**2 + Jm**2)")
plt.xlabel("Частота, Гц")
plt.ylabel("ln(Re**2 + Jm**2), у.е.")
plt.grid()


# График 4. Гистограмма амплитуд речевого сигнала

plt.figure()
plt.hist(signal, bins=50, color="steelblue")
plt.title("Гистограмма амплитуд речевого сигнала")
plt.xlabel("Амплитуда, у.е.")
plt.ylabel("Количество отсчётов")
plt.grid()


print("Графики построены. Закройте окна графиков, чтобы завершить программу.")
plt.show()
