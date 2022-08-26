# A decorator function that takes care of starting a coroutine
# automatically on call.
from copy import deepcopy

ENABLED = 1
DISABLED = 0





def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start


class Coroutine(object):
    def __init__(self):
        self._targets=[]
        
    def __or__(self, rhs):
        self.send_to(rhs)
        try:
            rhs.coro_fun = self.coro_fun
        except AttributeError:
            pass
        try:
            rhs.process = self.process
        except AttributeError:
            pass
        return rhs

    @coroutine
    def __coro_passthrough(self):
        # a coroutine implementation that just passes the ensemble without modifying
        # this coroutine is intended to replace the class own definition if the class should
        # be disabled (but is kept in the pipe line).
        while True:
            try:
                ens = (yield)
            except GeneratorExit:
                break
            else:
                self.send(ens)
        self.close_coroutine()

    def disable(self):
        # override the class's corofun with a passthrough version
        self.coro_fun = self.__coro_passthrough()

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
