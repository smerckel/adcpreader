# A decorator function that takes care of starting a coroutine
# automatically on call.
def coroutine(func):
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start


class Coroutine(object):

    def __init__(self):
        self._targets=[]
        
    def send_to(self, target):
        self._targets.append(target)

    def send(self, x):
        for i, t in enumerate(self._targets):
            if i:
                t.coro_fun.send(x.copy()) # other ensembles, by value (as a copy)
            else:
                t.coro_fun.send(x) # first ensemble, by reference

    def close_coroutine(self):
        for t in self._targets:
            t.coro_fun.close()
