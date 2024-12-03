import typing as t
from ast import literal_eval
from ast import parse
from itertools import chain
from itertools import islice
from types import GeneratorType
from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template

def native_concat(values: t.Iterable[t.Any]) -> t.Optional[t.Any]:
    """Return a native Python type from the list of compiled nodes. If
    the result is a single node, its value is returned. Otherwise, the
    nodes are concatenated as strings. If the result can be parsed with
    :func:`ast.literal_eval`, the parsed value is returned. Otherwise,
    the string is returned.

    :param values: Iterable of outputs to concatenate.
    """
    head = next(iter(values), None)

    if head is None:
        return None
    elif next(iter(values), None) is None:
        return head
    else:
        buf = []
        for value in values:
            if isinstance(value, str):
                buf.append(value)
            else:
                buf.append(str(value))
        return ''.join(buf)

class NativeCodeGenerator(CodeGenerator):
    """A code generator which renders Python types by not adding
    ``str()`` around output nodes.
    """

class NativeEnvironment(Environment):
    """An environment that renders templates to native Python types."""
    code_generator_class = NativeCodeGenerator
    concat = staticmethod(native_concat)

class NativeTemplate(Template):
    environment_class = NativeEnvironment

    def render(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Render the template to produce a native Python type. If the
        result is a single node, its value is returned. Otherwise, the
        nodes are concatenated as strings. If the result can be parsed
        with :func:`ast.literal_eval`, the parsed value is returned.
        Otherwise, the string is returned.
        """
        vars = dict(*args, **kwargs)
        try:
            result = self.root_render_func(self.new_context(vars))
            return native_concat(result)
        except Exception:
            exc_info = sys.exc_info()
            return self.environment.handle_exception(exc_info, True)
NativeEnvironment.template_class = NativeTemplate
