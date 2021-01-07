import heapq


def tensor_find(tensor, tensorSize, index):
    remaining = index
    loc = []
    for i in range(-1, -len(tensorSize) - 1, -1):
        size = tensorSize[i]
        loc.insert(0, remaining % size)
        remaining = int(remaining / size)

    val = tensor
    for l in loc:
        val = val[l]

    return val, loc

def plus_minus(x):
  return 1 if x > 0 else -1


class Gradients:
    def __init__(self, params1, params2):
        self.grads = {key: params1[key] - params2[key] for key in params1.keys()}

    def topN(self, N):
        abs_top_N = []

        for key in self.grads.keys():
            size = self.grads[key].size()
            total_size = 1
            for num in size:
                total_size *= num

            for i in range(total_size):
                val, loc = tensor_find(self.grads[key], size, i)
                if len(abs_top_N) < N:
                    heapq.heappush(abs_top_N, [abs(val), key, loc, plus_minus(val)])

                elif abs_top_N[0][0] > abs(val):
                    heapq.heapreplace(abs_top_N, [abs(val), key, loc, plus_minus(val)])

        return abs_top_N


class Parameters:
    def __init__(self, params):
        self.params = params

    def update_params(self, update_info):
        for grad in update_info:
            val = self.params[grad[1]]
            for l in grad[2]:
                val = val[l]
            val += grad[0]*grad[3]