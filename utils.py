import re
import sys
import socket
import hashlib

from calibre.utils.logging import Log

from .lib.cssselect import GenericTranslator, SelectorError


ns = {'x': 'http://www.w3.org/1999/xhtml'}
log = Log()
is_test = 'unittest' in sys.modules


def sep(char='=', count=30):
    return char * count


def css(seletor):
    try:
        return GenericTranslator().css_to_xpath(seletor, prefix='self::x:')
    except SelectorError:
        return None


def uid(*args):
    md5 = hashlib.md5()
    for arg in args:
        md5.update(arg if isinstance(arg, bytes) else arg.encode('utf-8'))
    return md5.hexdigest()


def trim(text):
    # Remove \xa0 to be compitable with Python2.x
    text = re.sub(u'\u00a0|\u3000', ' ', text)
    text = re.sub(u'\u200b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk(items, length=0):
    if length < 1:
        for item in items:
            yield [item]
        return
    item_length = len(items)
    length = item_length if length > item_length else length
    chunk_size = item_length / length
    for i in range(length):
        yield items[int(chunk_size*i):int(chunk_size*(i+1))]


def group(numbers):
    ranges = []
    current_range = []
    numbers = sorted(numbers)
    for number in numbers:
        if not current_range:
            current_range = [number, number]
        elif number - current_range[-1] == 1:
            current_range[-1] = number
        else:
            ranges.append(tuple(current_range))
            current_range = [number, number]
    ranges.append(tuple(current_range))
    return ranges


def sorted_mixed_keys(s):
    # https://docs.python.org/3/reference/expressions.html#value-comparisons
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', s)]


def is_str(data):
    return type(data).__name__ in ('str', 'unicode')


def is_proxy_availiable(host, port, timeout=1):
    try:
        host = host.replace('http://', '')
        socket.create_connection((host, int(port)), timeout).close()
    except Exception as e:
        return False
    return True


def dummy(*args, **kwargs):
    pass
