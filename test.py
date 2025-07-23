from torch.autograd.profiler import emit_itt
from yaml import emit

word = str(input())
print(word)

# Map: Tiền xử lý dữ liệu
# Map: Nhâận dữ liều đầu vào, xứ lý và chuyển thành các cặp key-value


def map(line):
    for word in line.strip().split():
        print(f"{word}\t1")
        #emit(word, 1)  # Emit the word with count 1

def reduce(key, values):
    total = 0
    for value in values:
        total += int(value)
        #emit(word,sum(value))
    print(f"{key}\t{total}")
#
map(word)

#reduce(word, [1])  # Example reduce call with a single value