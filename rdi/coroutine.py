# A decorator function that takes care of starting a coroutine
# automatically on call.

from copy import deepcopy

def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start


class Coroutine(object):

    def __init__(self):
        self._targets=[]
        
    def send_to(self, *targets):
        for target in targets:
            self._targets.append(target.coro_fun)

    def send(self, x):
        for i, coro in enumerate(self._targets):
            if i:
                coro.send(deepcopy(x)) # other ensembles, by value (as a copy)
            else:
                coro.send(x) # first ensemble, by reference

    def close_coroutine(self):
        for coro in self._targets:
            coro.close()
