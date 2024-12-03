"""Parse tokens from the lexer into nodes for the compiler."""
import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
if t.TYPE_CHECKING:
    import typing_extensions as te
    from .environment import Environment
_ImportInclude = t.TypeVar('_ImportInclude', nodes.Import, nodes.Include)
_MacroCall = t.TypeVar('_MacroCall', nodes.Macro, nodes.CallBlock)
_statement_keywords = frozenset(['for', 'if', 'block', 'extends', 'print', 'macro', 'include', 'from', 'import', 'set', 'with', 'autoescape'])
_compare_operators = frozenset(['eq', 'ne', 'lt', 'lteq', 'gt', 'gteq'])
_math_nodes: t.Dict[str, t.Type[nodes.Expr]] = {'add': nodes.Add, 'sub': nodes.Sub, 'mul': nodes.Mul, 'div': nodes.Div, 'floordiv': nodes.FloorDiv, 'mod': nodes.Mod}

class Parser:
    """This is the central parsing class Jinja uses.  It's passed to
    extensions and can be used to parse expressions or statements.
    """

    def __init__(self, environment: 'Environment', source: str, name: t.Optional[str]=None, filename: t.Optional[str]=None, state: t.Optional[str]=None) -> None:
        self.environment = environment
        self.stream = environment._tokenize(source, name, filename, state)
        self.name = name
        self.filename = filename
        self.closed = False
        self.extensions: t.Dict[str, t.Callable[['Parser'], t.Union[nodes.Node, t.List[nodes.Node]]]] = {}
        for extension in environment.iter_extensions():
            for tag in extension.tags:
                self.extensions[tag] = extension.parse
        self._last_identifier = 0
        self._tag_stack: t.List[str] = []
        self._end_token_stack: t.List[t.Tuple[str, ...]] = []

    def fail(self, msg: str, lineno: t.Optional[int]=None, exc: t.Type[TemplateSyntaxError]=TemplateSyntaxError) -> 'te.NoReturn':
        """Convenience method that raises `exc` with the message, passed
        line number or last line number as well as the current name and
        filename.
        """
        if lineno is None:
            lineno = self.stream.current.lineno
        raise exc(msg, lineno, self.name, self.filename)

    def fail_unknown_tag(self, name: str, lineno: t.Optional[int]=None) -> 'te.NoReturn':
        """Called if the parser encounters an unknown tag.  Tries to fail
        with a human readable error message that could help to identify
        the problem.
        """
        if lineno is None:
            lineno = self.stream.current.lineno
        if name == 'endif':
            self.fail('Encountered unexpected endif. Are you missing an opening "if" tag?', lineno)
        elif name == 'endfor':
            self.fail('Encountered unexpected endfor. Are you missing an opening "for" tag?', lineno)
        elif name == 'endblock':
            self.fail('Encountered unexpected endblock. Are you missing an opening "block" tag?', lineno)
        elif name in ('else', 'elif', 'elseif'):
            self.fail(f'Encountered unexpected {name}. Are you missing an opening "if" tag?', lineno)
        self.fail(f'Unknown tag {name!r}', lineno)

    def fail_eof(self, end_tokens: t.Optional[t.Tuple[str, ...]]=None, lineno: t.Optional[int]=None) -> 'te.NoReturn':
        """Like fail_unknown_tag but for end of template situations."""
        if end_tokens is not None:
            expected = ' or '.join(repr(x) for x in end_tokens)
            msg = f'Unexpected end of template. Expected {expected}.'
        else:
            msg = 'Unexpected end of template.'
        self.fail(msg, lineno)

    def is_tuple_end(self, extra_end_rules: t.Optional[t.Tuple[str, ...]]=None) -> bool:
        """Are we at the end of a tuple?"""
        if self.stream.current.type in ('variable_end', 'block_end', 'rparen'):
            return True
        if extra_end_rules is not None:
            return self.stream.current.test_any(extra_end_rules)
        return False

    def free_identifier(self, lineno: t.Optional[int]=None) -> nodes.InternalName:
        """Return a new free identifier as :class:`~jinja2.nodes.InternalName`."""
        self._last_identifier += 1
        rv = object.__new__(nodes.InternalName)
        rv.name = f'fi{self._last_identifier}'
        rv.lineno = lineno
        return rv

    def parse_statement(self) -> t.Union[nodes.Node, t.List[nodes.Node]]:
        """Parse a single statement."""
        token = self.stream.current
        if token.type != 'name':
            return self.parse_expression()
        if token.value in _statement_keywords:
            return getattr(self, f'parse_{token.value}')()
        if token.value == 'call':
            return self.parse_call_block()
        if token.value == 'filter':
            return self.parse_filter_block()
        return self.parse_expression()

    def parse_statements(self, end_tokens: t.Tuple[str, ...], drop_needle: bool=False) -> t.List[nodes.Node]:
        """Parse multiple statements into a list until one of the end tokens
        is reached.  This is used to parse the body of statements as it also
        parses template data if appropriate.  The parser checks first if the
        current token is a colon and skips it if there is one.  Then it checks
        for the block end and parses until if one of the `end_tokens` is
        reached.  Per default the active token in the stream at the end of
        the call is the matched end token.  If this is not wanted `drop_needle`
        can be set to `True` and the end token is removed.
        """
        result = []
        while 1:
            if self.stream.current.type == 'data':
                result.append(nodes.Output([nodes.TemplateData(self.stream.current.value)]))
                self.stream.next()
            elif self.stream.current.type == 'variable_begin':
                result.append(self.parse_tuple())
            elif self.stream.current.type == 'block_begin':
                self.stream.next()
                if self.stream.current.test_any(end_tokens):
                    if drop_needle:
                        self.stream.next()
                    return result
                result.append(self.parse_statement())
            else:
                break
        return result

    def parse_set(self) -> t.Union[nodes.Assign, nodes.AssignBlock]:
        """Parse an assign statement."""
        lineno = next(self.stream).lineno
        target = self.parse_assign_target()
        if self.stream.skip_if('assign'):
            expr = self.parse_expression()
            return nodes.Assign(target, expr, lineno=lineno)
        body = self.parse_statements(('name:endset',), drop_needle=True)
        return nodes.AssignBlock(target, body, lineno=lineno)

    def parse_for(self) -> nodes.For:
        """Parse a for loop."""
        lineno = next(self.stream).lineno
        target = self.parse_assign_target(extra_end_rules=('name:in',))
        self.stream.expect('name:in')
        iter = self.parse_expression()
        body = self.parse_statements(('name:endfor', 'name:else'))
        if next(self.stream).value == 'endfor':
            else_ = []
        else:
            else_ = self.parse_statements(('name:endfor',), drop_needle=True)
        return nodes.For(target, iter, body, else_, lineno=lineno)

    def parse_if(self) -> nodes.If:
        """Parse an if construct."""
        lineno = next(self.stream).lineno
        expr = self.parse_expression()
        body = self.parse_statements(('name:elif', 'name:else', 'name:endif'))
        elif_ = []
        else_ = []
        while 1:
            token = next(self.stream)
            if token.test('name:elif'):
                elif_.append((self.parse_expression(), self.parse_statements(('name:elif', 'name:else', 'name:endif'))))
            elif token.test('name:else'):
                else_ = self.parse_statements(('name:endif',), drop_needle=True)
                break
            else:
                break
        return nodes.If(expr, body, elif_, else_, lineno=lineno)

    def parse_assign_target(self, with_tuple: bool=True, name_only: bool=False, extra_end_rules: t.Optional[t.Tuple[str, ...]]=None, with_namespace: bool=False) -> t.Union[nodes.NSRef, nodes.Name, nodes.Tuple]:
        """Parse an assignment target.  As Jinja allows assignments to
        tuples, this function can parse all allowed assignment targets.  Per
        default assignments to tuples are parsed, that can be disable however
        by setting `with_tuple` to `False`.  If only assignments to names are
        wanted `name_only` can be set to `True`.  The `extra_end_rules`
        parameter is forwarded to the tuple parsing function.  If
        `with_namespace` is enabled, a namespace assignment may be parsed.
        """
        if name_only:
            token = self.stream.expect('name')
            return nodes.Name(token.value, 'store', lineno=token.lineno)
        if with_namespace:
            if self.stream.current.type == 'name' and self.stream.look().type == 'dot':
                namespace = self.stream.current.value
                self.stream.skip(2)
                token = self.stream.expect('name')
                return nodes.NSRef(namespace, token.value, lineno=token.lineno)
        if with_tuple:
            return self.parse_tuple(simplified=True, extra_end_rules=extra_end_rules)
        token = self.stream.expect('name')
        return nodes.Name(token.value, 'store', lineno=token.lineno)

    def parse_expression(self, with_condexpr: bool=True) -> nodes.Expr:
        """Parse an expression.  Per default all expressions are parsed, if
        the optional `with_condexpr` parameter is set to `False` conditional
        expressions are not parsed.
        """
        if with_condexpr:
            return self.parse_condexpr()
        return self.parse_or()

    def parse_tuple(self, simplified: bool=False, with_condexpr: bool=True, extra_end_rules: t.Optional[t.Tuple[str, ...]]=None, explicit_parentheses: bool=False) -> t.Union[nodes.Tuple, nodes.Expr]:
        """Works like `parse_expression` but if multiple expressions are
        delimited by a comma a :class:`~jinja2.nodes.Tuple` node is created.
        This method could also return a regular expression instead of a tuple
        if no commas where found.

        The default parsing mode is a full tuple.  If `simplified` is `True`
        only names and literals are parsed.  The `no_condexpr` parameter is
        forwarded to :meth:`parse_expression`.

        Because tuples do not require delimiters and may end in a bogus comma
        an extra hint is needed that marks the end of a tuple.  For example
        for loops support tuples between `for` and `in`.  In that case the
        `extra_end_rules` is set to ``['name:in']``.

        `explicit_parentheses` is true if the parsing was triggered by an
        expression in parentheses.  This is used to figure out if an empty
        tuple is a valid expression or not.
        """
        lineno = self.stream.current.lineno
        if simplified:
            parse = self.parse_primary
        elif with_condexpr:
            parse = self.parse_expression
        else:
            parse = lambda: self.parse_expression(with_condexpr=False)
        items = []
        is_tuple = False
        while 1:
            if self.is_tuple_end(extra_end_rules):
                break
            if simplified and self.stream.current.type == 'name':
                items.append(nodes.Name(self.stream.current.value, 'load'))
                self.stream.next()
            else:
                items.append(parse())
            if self.stream.current.type == 'comma':
                is_tuple = True
                self.stream.next()
            else:
                break
        if not is_tuple:
            if items:
                return items[0]
            if explicit_parentheses:
                return nodes.Tuple([], 'load', lineno=lineno)
        return nodes.Tuple(items, 'load', lineno=lineno)

    def parse(self) -> nodes.Template:
        """Parse the whole template into a `Template` node."""
        result = []
        while self.stream:
            token = self.stream.current
            if token.type == 'data':
                result.append(nodes.Output([nodes.TemplateData(token.value)]))
                self.stream.next()
            elif token.type == 'variable_begin':
                result.append(self.parse_tuple())
            elif token.type == 'block_begin':
                self.stream.next()
                result.append(self.parse_statement())
            else:
                break
        return nodes.Template(result, lineno=1)
