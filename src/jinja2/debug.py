import sys
import typing as t
from types import CodeType
from types import TracebackType
from .exceptions import TemplateSyntaxError
from .utils import internal_code
from .utils import missing
if t.TYPE_CHECKING:
    from .runtime import Context

def rewrite_traceback_stack(source: t.Optional[str]=None) -> BaseException:
    """Rewrite the current exception to replace any tracebacks from
    within compiled template code with tracebacks that look like they
    came from the template source.

    This must be called within an ``except`` block.

    :param source: For ``TemplateSyntaxError``, the original source if
        known.
    :return: The original exception with the rewritten traceback.
    """
    exc_type, exc_value, tb = sys.exc_info()
    if isinstance(exc_value, TemplateSyntaxError) and source is not None:
        exc_value.source = source

    new_tb = None
    while tb is not None:
        if tb.tb_frame.f_code in internal_code:
            template = tb.tb_frame.f_globals.get('__jinja_template__')
            if template is not None:
                lineno = template.get_corresponding_lineno(tb.tb_lineno)
                filename = template.filename or '<unknown>'
                new_tb = fake_traceback(
                    exc_value,
                    tb,
                    filename,
                    lineno
                )
                break
        tb = tb.tb_next

    if new_tb is None:
        return exc_value

    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next

    tb.tb_next = new_tb
    return exc_value

def fake_traceback(exc_value: BaseException, tb: t.Optional[TracebackType], filename: str, lineno: int) -> TracebackType:
    """Produce a new traceback object that looks like it came from the
    template source instead of the compiled code. The filename, line
    number, and location name will point to the template, and the local
    variables will be the current template context.

    :param exc_value: The original exception to be re-raised to create
        the new traceback.
    :param tb: The original traceback to get the local variables and
        code info from.
    :param filename: The template filename.
    :param lineno: The line number in the template source.
    """
    if tb is None:
        raise exc_value

    code = tb.tb_frame.f_code
    locals = get_template_locals(tb.tb_frame.f_locals)

    # Create a new code object with the updated filename and line number
    new_code = CodeType(
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        code.co_consts,
        code.co_names,
        code.co_varnames,
        filename,
        code.co_name,
        lineno,
        code.co_lnotab,
        code.co_freevars,
        code.co_cellvars
    )

    # Create a fake frame using the new code object and locals
    fake_frame = tb.tb_frame.__class__(new_code, tb.tb_frame.f_globals, locals)
    fake_frame.f_lineno = lineno

    # Create a new traceback object with the fake frame
    return TracebackType(
        tb_next=tb.tb_next,
        tb_frame=fake_frame,
        tb_lasti=tb.tb_lasti,
        tb_lineno=lineno
    )

def get_template_locals(real_locals: t.Mapping[str, t.Any]) -> t.Dict[str, t.Any]:
    """Based on the runtime locals, get the context that would be
    available at that point in the template.
    """
    ctx = real_locals.get('context')
    if ctx is not None:
        return ctx.get_all()
    
    return {
        k: v
        for k, v in real_locals.items()
        if not k.startswith('l_') and k != 'context'
    }
