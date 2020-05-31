import logging
import functools

def use_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwagrs):
        logging.warn('{} is running'.format(func.__name__))
        return func(*args, **kwagrs)
    return wrapper

@use_logging
def bar():
    print('i am bar')

@use_logging
def foo():
    print('i am foo')

if __name__ == '__main__':
    bar()
    foo()
    print(foo.__name__)