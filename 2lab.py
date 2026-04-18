import wave
import numpy as np
import matplotlib.pyplot as plt
import time

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
wav_file.close()

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

# Ввод количества отсчётов для всех графиков
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

print("Будет отображено", n, "отсчётов на всех графиках.")
print()

# Начало измерения времени вычислений (после всех вводов)
start_time = time.time()

# Обрезаем сигнал до n отсчётов — для всех графиков
signal = signal_full[:n]

# Дискретное время в секундах
T = 1 / framerate
time_axis = np.linspace(0.0, n * T, n, endpoint=False)


# График 1. Линейный сплошной с маркерами

plt.figure()
plt.plot(time_axis, signal, color="blue", linestyle="-", linewidth=1, marker="*", markersize=2)
plt.title("Линейный график речевого сигнала с маркерами '*'")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, у.е.")
plt.grid()


# График 2. Осциллограмма речевого сигнала

plt.figure()
plt.plot(time_axis, signal, color="blue", linestyle="-", linewidth=1)
plt.title("Осциллограмма речевого сигнала")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, у.е.")
plt.grid()


# График 3. Спектр: Мнимая часть ДПФ (Jm)

spectrum = np.fft.rfft(signal)
im = np.imag(spectrum)  # мнимая часть ДПФ
freqs = np.fft.rfftfreq(len(signal), T)

plt.figure()
plt.plot(freqs, im, color="green", linestyle="-", linewidth=1)
plt.title("Спектр: мнимая часть ДПФ (Jm)")
plt.xlabel("Частота, Гц")
plt.ylabel("Jm, у.е.")
plt.grid()


# График 4. Гистограмма амплитуд речевого сигнала

plt.figure()
plt.hist(signal, bins=50, color="steelblue")
plt.title("Гистограмма амплитуд речевого сигнала")
plt.xlabel("Амплитуда, у.е.")
plt.ylabel("Количество отсчётов")
plt.grid()


# Вычисление времени выполнения программы
execution_time = time.time() - start_time
print("Время выполнения программы:", round(execution_time, 4), "секунд")
print("Графики построены. Закройте окна графиков, чтобы завершить программу.")
plt.show()