import itertools
import json
import os
import time
from collections import namedtuple, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
from loguru import logger

TShirt = namedtuple('TShirt', ['name', 'min', 'max'])

with open('t_shirt_sizes.json') as f:
    t_shirt_sizes = json.load(f)['t_shirt_sizes']
    t_shirt_sizes = [TShirt(t_shirt['name'], t_shirt['min'], t_shirt['max']) for t_shirt in t_shirt_sizes]
    sorted(t_shirt_sizes, key=lambda x: x.min)
    t_shirt_by_name = {t_shirt.name: t_shirt for t_shirt in t_shirt_sizes}

XXXS = t_shirt_by_name['3XS']
XXS = t_shirt_by_name['XXS']
XS = t_shirt_by_name['XS']
S = t_shirt_by_name['S']
M = t_shirt_by_name['M']
L = t_shirt_by_name['L']
XL = t_shirt_by_name['XL']


def prob_x(t_shirt: TShirt, x: int) -> float:
    return 1.0 if t_shirt.min <= x and x <= t_shirt.max else 0.0


def t_shirts_from_name(*names: [str]) -> [TShirt]:
    return [t_shirt_by_name[name] for name in names]


def better_count(t_shirts: [TShirt]) -> list[tuple[int, int]]:
    min_x = sum(t_shirt.min for t_shirt in t_shirts)
    max_x = sum(t_shirt.max for t_shirt in t_shirts)
    result = [0 for _ in range(min_x, max_x + 1)]

    if len(t_shirts) > 2:
        pre_result = better_count(t_shirts[1:])
        current = t_shirts[0]
        for i in range(current.min, current.max + 1):
            for k, v in pre_result:
                print(f'k={k}, v={v}')
                index = i + k - min_x
                result[index] = v
    else:
        for t in itertools.product(*[list(range(t_shirt.min, t_shirt.max + 1)) for t_shirt in t_shirts]):
            result[sum(t) - min_x] += 1

    return list(zip(range(min_x, max_x + 1), result))


def naive_count(t_shirts: [TShirt]) -> list[tuple[int, int]]:
    min_x = sum(t_shirt.min for t_shirt in t_shirts)
    max_x = sum(t_shirt.max for t_shirt in t_shirts)
    result = [0 for _ in range(min_x, max_x + 1)]

    for t in itertools.product(*[list(range(t_shirt.min, t_shirt.max + 1)) for t_shirt in t_shirts]):
        result[sum(t) - min_x] += 1

    return list(zip(range(min_x, max_x + 1), result))


def efficient_count(t_shirts):
    min_x = sum(t_shirt.min for t_shirt in t_shirts)
    max_x = sum(t_shirt.max for t_shirt in t_shirts)

    def count_combinations(current_index, current_sum):
        if current_index == len(t_shirts):
            return 1 if current_sum == 0 else 0

        if (current_index, current_sum) in memo:
            return memo[(current_index, current_sum)]

        t_shirt = t_shirts[current_index]
        possibilities = 0
        for size in range(t_shirt.min, t_shirt.max + 1):
            possibilities += count_combinations(current_index + 1, current_sum - size)

        memo[(current_index, current_sum)] = possibilities
        return possibilities

    memo = {}
    possibilities = defaultdict(int)

    for i in range(min_x, max_x + 1):
        possibilities[i] = count_combinations(0, i)

    result = list(possibilities.items())
    return result


def plot(naive: [tuple[int, int]], better: [tuple[int, int]]):
    nxs = [naive[0][0] - 1] + [t[0] for t in naive] + [naive[-1][0] + 1]
    nys = [0] + [t[1] for t in naive] + [0]

    bxs = [better[0][0] - 1] + [t[0] for t in better] + [better[-1][0] + 1]
    bys = [0] + [t[1] for t in better] + [0]

    # Generate two normal distributions
    plt.plot(nxs, nys)
    plt.plot(bxs, bys, color='red')
    plt.xlim(min(nxs), max(nxs))  # Setze die x-Achse auf den gewünschten Bereich
    plt.ylim(0, max(nys) * 1.1)
    plt.show()


def format_result(data: list[tuple[int, int]]) -> str:
    return '{' + ', '.join(f'{k:02d}:{v:02d}' for k, v in data) + '}'


def visual_main():
    etappe = [
        XXXS, XXXS,
        XXS, XXS, XXS,
        XS,
        S,
        M, M, M,
        L,
        XL
    ]
    print(etappe)
    start = time.time()
    better_data = efficient_count(etappe)
    end = time.time()

    better_duration = int((end - start) * 1000)

    start = time.time()
    naive_data = efficient_count(etappe)
    end = time.time()

    naive_duration = int((end - start) * 1000)
    speedup = naive_duration / better_duration if naive_duration > 0 else None

    print(f'Calculated in {naive_duration}ms vs {better_duration}ms. Speedup {speedup}')
    print(format_result(naive_data))
    print(format_result(better_data))
    plot(naive_data, better_data)

    print()
    # result = prob_foo2(t_shirts_from_name(*['3XS', '3XS', 'XXS']))

    # print(result)


def save_lut(etappe):
    result = efficient_count(etappe) if len(etappe) > 0 else [(0, 0)]
    xs = [x for x, y in result]
    ys = [y for x, y in result]

    metadata = [
        etappe.count(XXXS),
        etappe.count(XXS),
        etappe.count(XS),
        etappe.count(S),
        etappe.count(M),
        etappe.count(L),
        etappe.count(XL),
    ]

    # row = metadata + ys[:len(ys) // 2]
    row = ys[:len(ys) // 2 + 1]

    #filename = '-'.join(list(str(x) for x in metadata))

    directory = os.path.join('dist', 'luts', '/'.join(list(str(x) for x in metadata)))
    path = os.path.join(directory, f'{min(xs)}_{max(xs)}.lut')

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, 'w') as f:
        f.write('\n'.join(list(str(x) for x in row)))


def main():
    limit = 600

    todo_count = 1
    search_space = []
    for t_shirt in t_shirt_sizes:
        search_space.append(list(range(0, max(1, limit // t_shirt.min))))
        todo_count *= max(1, limit // t_shirt.min)

    done = 0
    log_intervall = 10
    last_log = 0
    start_ts = time.time()
    for counts in itertools.product(*search_space):
        etappe = []
        for count, t_shirt in zip(counts, t_shirt_sizes):
            for i in range(count):
                etappe.append(t_shirt)
        save_lut(etappe)
        done += 1

        if time.time() - last_log > log_intervall:
            elapsed = time.time() - start_ts
            percentage = done / todo_count
            eta = elapsed * todo_count / done + start_ts
            eta = datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M:%S')

            logger.debug(f'{done / todo_count * 100:0.2f}%\t({done}/{todo_count})\t{int(done/elapsed)} #/s\tETA: {eta}')
            last_log = time.time()


if __name__ == '__main__':
    main()
