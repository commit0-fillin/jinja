"""Extension API for adding custom tags and behavior."""
import pprint
import re
import typing as t
from markupsafe import Markup
from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context
if t.TYPE_CHECKING:
    import typing_extensions as te
    from .lexer import Token
    from .lexer import TokenStream
    from .parser import Parser

    class _TranslationsBasic(te.Protocol):
        pass

    class _TranslationsContext(_TranslationsBasic):
        pass
    _SupportedTranslations = t.Union[_TranslationsBasic, _TranslationsContext]
GETTEXT_FUNCTIONS: t.Tuple[str, ...] = ('_', 'gettext', 'ngettext', 'pgettext', 'npgettext')
_ws_re = re.compile('\\s*\\n\\s*')

class Extension:
    """Extensions can be used to add extra functionality to the Jinja template
    system at the parser level.  Custom extensions are bound to an environment
    but may not store environment specific data on `self`.  The reason for
    this is that an extension can be bound to another environment (for
    overlays) by creating a copy and reassigning the `environment` attribute.

    As extensions are created by the environment they cannot accept any
    arguments for configuration.  One may want to work around that by using
    a factory function, but that is not possible as extensions are identified
    by their import name.  The correct way to configure the extension is
    storing the configuration values on the environment.  Because this way the
    environment ends up acting as central configuration storage the
    attributes may clash which is why extensions have to ensure that the names
    they choose for configuration are not too generic.  ``prefix`` for example
    is a terrible name, ``fragment_cache_prefix`` on the other hand is a good
    name as includes the name of the extension (fragment cache).
    """
    identifier: t.ClassVar[str]

    def __init_subclass__(cls) -> None:
        cls.identifier = f'{cls.__module__}.{cls.__name__}'
    tags: t.Set[str] = set()
    priority = 100

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def bind(self, environment: Environment) -> 'Extension':
        """Create a copy of this extension bound to another environment."""
        rv = object.__new__(self.__class__)
        rv.__dict__.update(self.__dict__)
        rv.environment = environment
        return rv

    def preprocess(self, source: str, name: t.Optional[str], filename: t.Optional[str]=None) -> str:
        """This method is called before the actual lexing and can be used to
        preprocess the source.  The `filename` is optional.  The return value
        must be the preprocessed source.
        """
        return source

    def filter_stream(self, stream: 'TokenStream') -> t.Union['TokenStream', t.Iterable['Token']]:
        """It's passed a :class:`~jinja2.lexer.TokenStream` that can be used
        to filter tokens returned.  This method has to return an iterable of
        :class:`~jinja2.lexer.Token`\\s, but it doesn't have to return a
        :class:`~jinja2.lexer.TokenStream`.
        """
        return stream

    def parse(self, parser: 'Parser') -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """If any of the :attr:`tags` matched this method is called with the
        parser as first argument.  The token the parser stream is pointing at
        is the name token that matched.  This method has to return one or a
        list of multiple nodes.
        """
        raise NotImplementedError(f'Extension {self.__class__.__name__} does not implement parse method')

    def attr(self, name: str, lineno: t.Optional[int]=None) -> nodes.ExtensionAttribute:
        """Return an attribute node for the current extension.  This is useful
        to pass constants on extensions to generated template code.

        ::

            self.attr('_my_attribute', lineno=lineno)
        """
        return nodes.ExtensionAttribute(self.identifier, name, lineno=lineno)

    def call_method(self, name: str, args: t.Optional[t.List[nodes.Expr]]=None, kwargs: t.Optional[t.List[nodes.Keyword]]=None, dyn_args: t.Optional[nodes.Expr]=None, dyn_kwargs: t.Optional[nodes.Expr]=None, lineno: t.Optional[int]=None) -> nodes.Call:
        """Call a method of the extension.  This is a shortcut for
        :meth:`attr` + :class:`jinja2.nodes.Call`.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = []
        return nodes.Call(self.attr(name, lineno=lineno), args, kwargs, dyn_args, dyn_kwargs, lineno=lineno)

class InternationalizationExtension(Extension):
    """This extension adds gettext support to Jinja."""
    tags = {'trans'}

    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)
        environment.globals['_'] = _gettext_alias
        environment.extend(install_gettext_translations=self._install, install_null_translations=self._install_null, install_gettext_callables=self._install_callables, uninstall_gettext_translations=self._uninstall, extract_translations=self._extract, newstyle_gettext=False)

    def parse(self, parser: 'Parser') -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """Parse a translatable tag."""
        lineno = next(parser.stream).lineno
        token = parser.stream.current
        if token.type != 'name':
            parser.fail('Expected translation keyword', token.lineno)

        if token.value == 'trans':
            return self._parse_trans(parser)
        elif token.value == 'pluralize':
            return self._parse_pluralize(parser)
        else:
            parser.fail('Unknown translation keyword', token.lineno)

    def _parse_block(self, parser: 'Parser', allow_pluralize: bool) -> t.Tuple[t.List[str], str]:
        """Parse until the next block tag with a given name."""
        buf = []
        while 1:
            if parser.stream.current.type == 'data':
                buf.append(parser.stream.current.value)
                next(parser.stream)
            elif parser.stream.current.type == 'variable_begin':
                buf.append(self._parse_expression(parser))
            elif parser.stream.current.type == 'block_begin':
                if allow_pluralize:
                    if parser.stream.look().test('name:pluralize'):
                        return ''.join(buf), 'pluralize'
                if parser.stream.look().test('name:endtrans'):
                    return ''.join(buf), 'endtrans'
                parser.fail('Unknown tag in translation block')
            else:
                parser.fail('Unexpected token in translation block')

    def _make_node(self, singular: str, plural: t.Optional[str], context: t.Optional[str], variables: t.Dict[str, nodes.Expr], plural_expr: t.Optional[nodes.Expr], vars_referenced: bool, num_called_num: bool) -> nodes.Output:
        """Generates a useful node from the data provided."""
        func_name = 'gettext' if plural is None else 'ngettext'
        if context is not None:
            func_name = 'p' + func_name
        
        gettext_node = nodes.Name(func_name, 'load')
        
        args = [nodes.Const(singular)]
        if plural is not None:
            args.append(nodes.Const(plural))
        if context is not None:
            args.insert(0, nodes.Const(context))
        
        if plural_expr is not None:
            args.append(plural_expr)
        
        node = nodes.Call(gettext_node, args, [], None, None)
        
        if vars_referenced:
            node = nodes.Mod(node, nodes.Dict([
                nodes.Pair(nodes.Const(key), val)
                for key, val in variables.items()
            ]))
        
        return nodes.Output([node])

class ExprStmtExtension(Extension):
    """Adds a `do` tag to Jinja that works like the print statement just
    that it doesn't print the return value.
    """
    tags = {'do'}

class LoopControlExtension(Extension):
    """Adds break and continue to the template engine."""
    tags = {'break', 'continue'}

class DebugExtension(Extension):
    """A ``{% debug %}`` tag that dumps the available variables,
    filters, and tests.

    .. code-block:: html+jinja

        <pre>{% debug %}</pre>

    .. code-block:: text

        {'context': {'cycler': <class 'jinja2.utils.Cycler'>,
                     ...,
                     'namespace': <class 'jinja2.utils.Namespace'>},
         'filters': ['abs', 'attr', 'batch', 'capitalize', 'center', 'count', 'd',
                     ..., 'urlencode', 'urlize', 'wordcount', 'wordwrap', 'xmlattr'],
         'tests': ['!=', '<', '<=', '==', '>', '>=', 'callable', 'defined',
                   ..., 'odd', 'sameas', 'sequence', 'string', 'undefined', 'upper']}

    .. versionadded:: 2.11.0
    """
    tags = {'debug'}

def extract_from_ast(ast: nodes.Template, gettext_functions: t.Sequence[str]=GETTEXT_FUNCTIONS, babel_style: bool=True) -> t.Iterator[t.Tuple[int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]]]]:
    """Extract localizable strings from the given template node.  Per
    default this function returns matches in babel style that means non string
    parameters as well as keyword arguments are returned as `None`.  This
    allows Babel to figure out what you really meant if you are using
    gettext functions that allow keyword arguments for placeholder expansion.
    If you don't want that behavior set the `babel_style` parameter to `False`
    which causes only strings to be returned and parameters are always stored
    in tuples.  As a consequence invalid gettext calls (calls without a single
    string parameter or string parameters after non-string parameters) are
    skipped.

    This example explains the behavior:

    >>> from jinja2 import Environment
    >>> env = Environment()
    >>> node = env.parse('{{ (_("foo"), _(), ngettext("foo", "bar", 42)) }}')
    >>> list(extract_from_ast(node))
    [(1, '_', 'foo'), (1, '_', ()), (1, 'ngettext', ('foo', 'bar', None))]
    >>> list(extract_from_ast(node, babel_style=False))
    [(1, '_', ('foo',)), (1, 'ngettext', ('foo', 'bar'))]

    For every string found this function yields a ``(lineno, function,
    message)`` tuple, where:

    * ``lineno`` is the number of the line on which the string was found,
    * ``function`` is the name of the ``gettext`` function used (if the
      string was extracted from embedded Python code), and
    *   ``message`` is the string, or a tuple of strings for functions
         with multiple string arguments.

    This extraction function operates on the AST and is because of that unable
    to extract any comments.  For comment support you have to use the babel
    extraction interface or extract comments yourself.
    """
    for node in ast.find_all(nodes.Call):
        if not isinstance(node.node, nodes.Name) or \
           node.node.name not in gettext_functions:
            continue

        strings = []
        for arg in node.args:
            if isinstance(arg, nodes.Const) and \
               isinstance(arg.value, str):
                strings.append(arg.value)
            else:
                strings.append(None)

        for _ in range(len(strings), 3):
            strings.append(None)

        if not babel_style:
            strings = tuple(s for s in strings if s is not None)
            if not strings:
                continue

        yield node.lineno, node.node.name, \
            strings[0] if len(strings) == 1 else tuple(strings)

class _CommentFinder:
    """Helper class to find comments in a token stream.  Can only
    find comments for gettext calls forwards.  Once the comment
    from line 4 is found, a comment for line 1 will not return a
    usable value.
    """

    def __init__(self, tokens: t.Sequence[t.Tuple[int, str, str]], comment_tags: t.Sequence[str]) -> None:
        self.tokens = tokens
        self.comment_tags = comment_tags
        self.offset = 0
        self.last_lineno = 0

def babel_extract(fileobj: t.BinaryIO, keywords: t.Sequence[str], comment_tags: t.Sequence[str], options: t.Dict[str, t.Any]) -> t.Iterator[t.Tuple[int, str, t.Union[t.Optional[str], t.Tuple[t.Optional[str], ...]], t.List[str]]]:
    """Babel extraction method for Jinja templates.

    .. versionchanged:: 2.3
       Basic support for translation comments was added.  If `comment_tags`
       is now set to a list of keywords for extraction, the extractor will
       try to find the best preceding comment that begins with one of the
       keywords.  For best results, make sure to not have more than one
       gettext call in one line of code and the matching comment in the
       same line or the line before.

    .. versionchanged:: 2.5.1
       The `newstyle_gettext` flag can be set to `True` to enable newstyle
       gettext calls.

    .. versionchanged:: 2.7
       A `silent` option can now be provided.  If set to `False` template
       syntax errors are propagated instead of being ignored.

    :param fileobj: the file-like object the messages should be extracted from
    :param keywords: a list of keywords (i.e. function names) that should be
                     recognized as translation functions
    :param comment_tags: a list of translator tags to search for and include
                         in the results.
    :param options: a dictionary of additional options (optional)
    :return: an iterator over ``(lineno, funcname, message, comments)`` tuples.
             (comments will be empty currently)
    """
    from jinja2 import Environment, TemplateSyntaxError

    extensions = set(options.get('extensions', ()))
    silent = options.get('silent', True)
    newstyle = options.get('newstyle', False)

    if newstyle:
        extensions.add('jinja2.ext.do')

    env = Environment(extensions=extensions)
    
    try:
        source = fileobj.read().decode(options.get('encoding', 'utf-8'))
        node = env.parse(source)
        for lineno, func, message in extract_from_ast(node, keywords):
            yield lineno, func, message, []
    except TemplateSyntaxError as e:
        if not silent:
            raise
        yield e.lineno, None, f'Syntax error: {e}', []
i18n = InternationalizationExtension
do = ExprStmtExtension
loopcontrols = LoopControlExtension
debug = DebugExtension
