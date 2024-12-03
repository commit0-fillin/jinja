import enum
import json
import os
import re
import typing as t
from collections import abc
from collections import deque
from random import choice
from random import randrange
from threading import Lock
from types import CodeType
from urllib.parse import quote_from_bytes
import markupsafe
if t.TYPE_CHECKING:
    import typing_extensions as te
F = t.TypeVar('F', bound=t.Callable[..., t.Any])
missing: t.Any = type('MissingType', (), {'__repr__': lambda x: 'missing'})()
internal_code: t.MutableSet[CodeType] = set()
concat = ''.join

def pass_context(f: F) -> F:
    """Pass the :class:`~jinja2.runtime.Context` as the first argument
    to the decorated function when called while rendering a template.

    Can be used on functions, filters, and tests.

    If only ``Context.eval_context`` is needed, use
    :func:`pass_eval_context`. If only ``Context.environment`` is
    needed, use :func:`pass_environment`.

    .. versionadded:: 3.0.0
        Replaces ``contextfunction`` and ``contextfilter``.
    """
    f.pass_context = True
    return f

def pass_eval_context(f: F) -> F:
    """Pass the :class:`~jinja2.nodes.EvalContext` as the first argument
    to the decorated function when called while rendering a template.
    See :ref:`eval-context`.

    Can be used on functions, filters, and tests.

    If only ``EvalContext.environment`` is needed, use
    :func:`pass_environment`.

    .. versionadded:: 3.0.0
        Replaces ``evalcontextfunction`` and ``evalcontextfilter``.
    """
    f.pass_eval_context = True
    return f

def pass_environment(f: F) -> F:
    """Pass the :class:`~jinja2.Environment` as the first argument to
    the decorated function when called while rendering a template.

    Can be used on functions, filters, and tests.

    .. versionadded:: 3.0.0
        Replaces ``environmentfunction`` and ``environmentfilter``.
    """
    f.pass_environment = True
    return f

class _PassArg(enum.Enum):
    context = enum.auto()
    eval_context = enum.auto()
    environment = enum.auto()

def internalcode(f: F) -> F:
    """Marks the function as internally used"""
    f.internal = True
    return f

def is_undefined(obj: t.Any) -> bool:
    """Check if the object passed is undefined.  This does nothing more than
    performing an instance check against :class:`Undefined` but looks nicer.
    This can be used for custom filters or tests that want to react to
    undefined variables.  For example a custom default filter can look like
    this::

        def default(var, default=''):
            if is_undefined(var):
                return default
            return var
    """
    from .runtime import Undefined
    return isinstance(obj, Undefined)

def consume(iterable: t.Iterable[t.Any]) -> None:
    """Consumes an iterable without doing anything with it."""
    for _ in iterable:
        pass

def clear_caches() -> None:
    """Jinja keeps internal caches for environments and lexers.  These are
    used so that Jinja doesn't have to recreate environments and lexers all
    the time.  Normally you don't have to care about that but if you are
    measuring memory consumption you may want to clean the caches.
    """
    from .environment import get_spontaneous_environment
    get_spontaneous_environment.cache_clear()
    from .lexer import _lexer_cache
    _lexer_cache.clear()

def import_string(import_name: str, silent: bool=False) -> t.Any:
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If the `silent` is True the return value will be `None` if the import
    fails.

    :return: imported object
    """
    try:
        if ':' in import_name:
            module, obj = import_name.split(':', 1)
        elif '.' in import_name:
            module, obj = import_name.rsplit('.', 1)
        else:
            return __import__(import_name)
        return getattr(__import__(module, None, None, [obj]), obj)
    except (ImportError, AttributeError):
        if not silent:
            raise
    return None

def open_if_exists(filename: str, mode: str='rb') -> t.Optional[t.IO[t.Any]]:
    """Returns a file descriptor for the filename if that file exists,
    otherwise ``None``.
    """
    try:
        return open(filename, mode)
    except IOError:
        return None

def object_type_repr(obj: t.Any) -> str:
    """Returns the name of the object's type.  For some recognized
    singletons the name of the object is returned instead. (For
    example for `None` and `Ellipsis`).
    """
    if obj is None:
        return 'None'
    elif obj is Ellipsis:
        return 'Ellipsis'
    return obj.__class__.__name__

def pformat(obj: t.Any) -> str:
    """Format an object using :func:`pprint.pformat`."""
    from pprint import pformat
    return pformat(obj)
_http_re = re.compile('\n    ^\n    (\n        (https?://|www\\.)  # scheme or www\n        (([\\w%-]+\\.)+)?  # subdomain\n        (\n            [a-z]{2,63}  # basic tld\n        |\n            xn--[\\w%]{2,59}  # idna tld\n        )\n    |\n        ([\\w%-]{2,63}\\.)+  # basic domain\n        (com|net|int|edu|gov|org|info|mil)  # basic tld\n    |\n        (https?://)  # scheme\n        (\n            (([\\d]{1,3})(\\.[\\d]{1,3}){3})  # IPv4\n        |\n            (\\[([\\da-f]{0,4}:){2}([\\da-f]{0,4}:?){1,6}])  # IPv6\n        )\n    )\n    (?::[\\d]{1,5})?  # port\n    (?:[/?#]\\S*)?  # path, query, and fragment\n    $\n    ', re.IGNORECASE | re.VERBOSE)
_email_re = re.compile('^\\S+@\\w[\\w.-]*\\.\\w+$')

def urlize(text: str, trim_url_limit: t.Optional[int]=None, rel: t.Optional[str]=None, target: t.Optional[str]=None, extra_schemes: t.Optional[t.Iterable[str]]=None) -> str:
    """Convert URLs in text into clickable links.

    This may not recognize links in some situations. Usually, a more
    comprehensive formatter, such as a Markdown library, is a better
    choice.

    Works on ``http://``, ``https://``, ``www.``, ``mailto:``, and email
    addresses. Links with trailing punctuation (periods, commas, closing
    parentheses) and leading punctuation (opening parentheses) are
    recognized excluding the punctuation. Email addresses that include
    header fields are not recognized (for example,
    ``mailto:address@example.com?cc=copy@example.com``).

    :param text: Original text containing URLs to link.
    :param trim_url_limit: Shorten displayed URL values to this length.
    :param target: Add the ``target`` attribute to links.
    :param rel: Add the ``rel`` attribute to links.
    :param extra_schemes: Recognize URLs that start with these schemes
        in addition to the default behavior.

    .. versionchanged:: 3.0
        The ``extra_schemes`` parameter was added.

    .. versionchanged:: 3.0
        Generate ``https://`` links for URLs without a scheme.

    .. versionchanged:: 3.0
        The parsing rules were updated. Recognize email addresses with
        or without the ``mailto:`` scheme. Validate IP addresses. Ignore
        parentheses and brackets in more cases.
    """
    from markupsafe import escape
    import re

    schemes = ['http', 'https', 'mailto'] + (list(extra_schemes) if extra_schemes else [])
    words = re.split(r'(\s+)', str(escape(text)))

    for i, word in enumerate(words):
        if '.' in word or '@' in word or ':' in word:
            for scheme in schemes:
                if word.startswith(f'{scheme}:'):
                    break
            else:
                if word.startswith('www.') or '@' in word:
                    word = f'http://{word}'
            
            url = word
            if trim_url_limit and len(url) > trim_url_limit:
                url = f'{url[:trim_url_limit]}...'
            
            attrs = []
            if target:
                attrs.append(f'target="{escape(target)}"')
            if rel:
                attrs.append(f'rel="{escape(rel)}"')
            
            words[i] = f'<a href="{escape(word)}" {"".join(attrs)}>{escape(url)}</a>'

    return ''.join(words)

def generate_lorem_ipsum(n: int=5, html: bool=True, min: int=20, max: int=100) -> str:
    """Generate some lorem ipsum for the template."""
    from random import randrange
    
    lorem = '''
    Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
    veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
    commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
    velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
    occaecat cupidatat non proident, sunt in culpa qui officia deserunt
    mollit anim id est laborum.
    '''.strip()

    words = lorem.split()
    result = []

    for _ in range(n):
        next_capitalized = True
        last_comma = last_fullstop = 0
        word_count = 0
        last = None
        p = []

        # each paragraph contains out of 20 to 100 words.
        for idx, word in enumerate(words):
            if word_count >= min:
                if randrange(100) < 10:
                    last = ['.', '?', '!'][randrange(3)]
                    p.append(word + last)
                    break
                elif randrange(100) < 20:
                    last = ','
                    p.append(word + last)
                    if len(p) - last_comma > 3 or randrange(100) < 10:
                        next_capitalized = True
                else:
                    p.append(word)
            elif last == ',' and randrange(100) < 10:
                next_capitalized = True

            if next_capitalized:
                word = word.capitalize()
                next_capitalized = False

            last_word = word

            if word_count < max:
                p.append(word)
                word_count += 1

                if last == '.' or last == '?' or last == '!':
                    next_capitalized = True

        result.append(' '.join(p))

    if not html:
        return '\n\n'.join(result)
    return '\n'.join(f'<p>{paragraph}</p>' for paragraph in result)

def url_quote(obj: t.Any, charset: str='utf-8', for_qs: bool=False) -> str:
    """Quote a string for use in a URL using the given charset.

    :param obj: String or bytes to quote. Other types are converted to
        string then encoded to bytes using the given charset.
    :param charset: Encode text to bytes using this charset.
    :param for_qs: Quote "/" and use "+" for spaces.
    """
    from urllib.parse import quote
    
    if not isinstance(obj, (str, bytes)):
        obj = str(obj)
    if isinstance(obj, str):
        obj = obj.encode(charset)
    
    safe = b"" if for_qs else b"/"
    rv = quote(obj, safe=safe)
    if for_qs:
        rv = rv.replace('%20', '+')
    return rv

@abc.MutableMapping.register
class LRUCache:
    """A simple LRU Cache implementation."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._mapping: t.Dict[t.Any, t.Any] = {}
        self._queue: 'te.Deque[t.Any]' = deque()
        self._postinit()

    def __getstate__(self) -> t.Mapping[str, t.Any]:
        return {'capacity': self.capacity, '_mapping': self._mapping, '_queue': self._queue}

    def __setstate__(self, d: t.Mapping[str, t.Any]) -> None:
        self.__dict__.update(d)
        self._postinit()

    def __getnewargs__(self) -> t.Tuple[t.Any, ...]:
        return (self.capacity,)

    def copy(self) -> 'LRUCache':
        """Return a shallow copy of the instance."""
        rv = type(self)(self.capacity)
        rv._mapping = self._mapping.copy()
        rv._queue = self._queue.copy()
        return rv

    def get(self, key: t.Any, default: t.Any=None) -> t.Any:
        """Return an item from the cache dict or `default`"""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: t.Any, default: t.Any=None) -> t.Any:
        """Set `default` if the key is not in the cache otherwise
        leave unchanged. Return the value of this key.
        """
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def clear(self) -> None:
        """Clear the cache."""
        self._mapping.clear()
        self._queue.clear()

    def __contains__(self, key: t.Any) -> bool:
        """Check if a key exists in this cache."""
        return key in self._mapping

    def __len__(self) -> int:
        """Return the current size of the cache."""
        return len(self._mapping)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self._mapping!r}>'

    def __getitem__(self, key: t.Any) -> t.Any:
        """Get an item from the cache. Moves the item up so that it has the
        highest priority then.

        Raise a `KeyError` if it does not exist.
        """
        with self._wlock:
            rv = self._mapping[key]
            if self._queue[-1] != key:
                try:
                    self._remove(key)
                except ValueError:
                    pass
                self._append(key)
            return rv

    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        """Sets the value for an item. Moves the item up so that it
        has the highest priority then.
        """
        with self._wlock:
            if key in self._mapping:
                self._remove(key)
            elif len(self._mapping) == self.capacity:
                del self._mapping[self._popleft()]
            self._append(key)
            self._mapping[key] = value

    def __delitem__(self, key: t.Any) -> None:
        """Remove an item from the cache dict.
        Raise a `KeyError` if it does not exist.
        """
        with self._wlock:
            del self._mapping[key]
            try:
                self._remove(key)
            except ValueError:
                pass

    def items(self) -> t.Iterable[t.Tuple[t.Any, t.Any]]:
        """Return a list of items."""
        return self._mapping.items()

    def values(self) -> t.Iterable[t.Any]:
        """Return a list of all values."""
        return self._mapping.values()

    def keys(self) -> t.Iterable[t.Any]:
        """Return a list of all keys ordered by most recent usage."""
        return reversed(tuple(self._queue))

    def __iter__(self) -> t.Iterator[t.Any]:
        return reversed(tuple(self._queue))

    def __reversed__(self) -> t.Iterator[t.Any]:
        """Iterate over the keys in the cache dict, oldest items
        coming first.
        """
        return iter(tuple(self._queue))
    __copy__ = copy

def select_autoescape(enabled_extensions: t.Collection[str]=('html', 'htm', 'xml'), disabled_extensions: t.Collection[str]=(), default_for_string: bool=True, default: bool=False) -> t.Callable[[t.Optional[str]], bool]:
    """Intelligently sets the initial value of autoescaping based on the
    filename of the template.  This is the recommended way to configure
    autoescaping if you do not want to write a custom function yourself.

    If you want to enable it for all templates created from strings or
    for all templates with `.html` and `.xml` extensions::

        from jinja2 import Environment, select_autoescape
        env = Environment(autoescape=select_autoescape(
            enabled_extensions=('html', 'xml'),
            default_for_string=True,
        ))

    Example configuration to turn it on at all times except if the template
    ends with `.txt`::

        from jinja2 import Environment, select_autoescape
        env = Environment(autoescape=select_autoescape(
            disabled_extensions=('txt',),
            default_for_string=True,
            default=True,
        ))

    The `enabled_extensions` is an iterable of all the extensions that
    autoescaping should be enabled for.  Likewise `disabled_extensions` is
    a list of all templates it should be disabled for.  If a template is
    loaded from a string then the default from `default_for_string` is used.
    If nothing matches then the initial value of autoescaping is set to the
    value of `default`.

    For security reasons this function operates case insensitive.

    .. versionadded:: 2.9
    """
    enabled_extensions = tuple(x.lower() for x in enabled_extensions)
    disabled_extensions = tuple(x.lower() for x in disabled_extensions)

    def autoescape(template_name: t.Optional[str]) -> bool:
        if template_name is None:
            return default_for_string
        ext = template_name.rsplit('.', 1)[-1].lower()
        if ext in enabled_extensions:
            return True
        if ext in disabled_extensions:
            return False
        return default

    return autoescape

def htmlsafe_json_dumps(obj: t.Any, dumps: t.Optional[t.Callable[..., str]]=None, **kwargs: t.Any) -> markupsafe.Markup:
    """Serialize an object to a string of JSON with :func:`json.dumps`,
    then replace HTML-unsafe characters with Unicode escapes and mark
    the result safe with :class:`~markupsafe.Markup`.

    This is available in templates as the ``|tojson`` filter.

    The following characters are escaped: ``<``, ``>``, ``&``, ``'``.

    The returned string is safe to render in HTML documents and
    ``<script>`` tags. The exception is in HTML attributes that are
    double quoted; either use single quotes or the ``|forceescape``
    filter.

    :param obj: The object to serialize to JSON.
    :param dumps: The ``dumps`` function to use. Defaults to
        ``env.policies["json.dumps_function"]``, which defaults to
        :func:`json.dumps`.
    :param kwargs: Extra arguments to pass to ``dumps``. Merged onto
        ``env.policies["json.dumps_kwargs"]``.

    .. versionchanged:: 3.0
        The ``dumper`` parameter is renamed to ``dumps``.

    .. versionadded:: 2.9
    """
    if dumps is None:
        import json
        dumps = json.dumps

    return markupsafe.Markup(
        dumps(obj, **kwargs)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )

class Cycler:
    """Cycle through values by yield them one at a time, then restarting
    once the end is reached. Available as ``cycler`` in templates.

    Similar to ``loop.cycle``, but can be used outside loops or across
    multiple loops. For example, render a list of folders and files in a
    list, alternating giving them "odd" and "even" classes.

    .. code-block:: html+jinja

        {% set row_class = cycler("odd", "even") %}
        <ul class="browser">
        {% for folder in folders %}
          <li class="folder {{ row_class.next() }}">{{ folder }}
        {% endfor %}
        {% for file in files %}
          <li class="file {{ row_class.next() }}">{{ file }}
        {% endfor %}
        </ul>

    :param items: Each positional argument will be yielded in the order
        given for each cycle.

    .. versionadded:: 2.1
    """

    def __init__(self, *items: t.Any) -> None:
        if not items:
            raise RuntimeError('at least one item has to be provided')
        self.items = items
        self.pos = 0

    def reset(self) -> None:
        """Resets the current item to the first item."""
        pass

    @property
    def current(self) -> t.Any:
        """Return the current item. Equivalent to the item that will be
        returned next time :meth:`next` is called.
        """
        pass

    def next(self) -> t.Any:
        """Return the current item, then advance :attr:`current` to the
        next item.
        """
        pass
    __next__ = next

class Joiner:
    """A joining helper for templates."""

    def __init__(self, sep: str=', ') -> None:
        self.sep = sep
        self.used = False

    def __call__(self) -> str:
        if not self.used:
            self.used = True
            return ''
        return self.sep

class Namespace:
    """A namespace object that can hold arbitrary attributes.  It may be
    initialized from a dictionary or with keyword arguments."""

    def __init__(*args: t.Any, **kwargs: t.Any) -> None:
        self, args = (args[0], args[1:])
        self.__attrs = dict(*args, **kwargs)

    def __getattribute__(self, name: str) -> t.Any:
        if name in {'_Namespace__attrs', '__class__'}:
            return object.__getattribute__(self, name)
        try:
            return self.__attrs[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setitem__(self, name: str, value: t.Any) -> None:
        self.__attrs[name] = value

    def __repr__(self) -> str:
        return f'<Namespace {self.__attrs!r}>'
