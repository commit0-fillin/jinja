"""API and implementations for loading templates from different data
sources.
"""
import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType
from .exceptions import TemplateNotFound
from .utils import internalcode
if t.TYPE_CHECKING:
    from .environment import Environment
    from .environment import Template

def split_template_path(template: str) -> t.List[str]:
    """Split a path into segments and perform a sanity check.  If it detects
    '..' in the path it will raise a `TemplateNotFound` error.
    """
    pieces = []
    for piece in template.split('/'):
        if piece == '..':
            raise TemplateNotFound(template)
        elif piece and piece != '.':
            pieces.append(piece)
    return pieces

class BaseLoader:
    """Baseclass for all loaders.  Subclass this and override `get_source` to
    implement a custom loading mechanism.  The environment provides a
    `get_template` method that calls the loader's `load` method to get the
    :class:`Template` object.

    A very basic example for a loader that looks up templates on the file
    system could look like this::

        from jinja2 import BaseLoader, TemplateNotFound
        from os.path import join, exists, getmtime

        class MyLoader(BaseLoader):

            def __init__(self, path):
                self.path = path

            def get_source(self, environment, template):
                path = join(self.path, template)
                if not exists(path):
                    raise TemplateNotFound(template)
                mtime = getmtime(path)
                with open(path) as f:
                    source = f.read()
                return source, path, lambda: mtime == getmtime(path)
    """
    has_source_access = True

    def get_source(self, environment: 'Environment', template: str) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        """Get the template source, filename and reload helper for a template.
        It's passed the environment and template name and has to return a
        tuple in the form ``(source, filename, uptodate)`` or raise a
        `TemplateNotFound` error if it can't locate the template.

        The source part of the returned tuple must be the source of the
        template as a string. The filename should be the name of the
        file on the filesystem if it was loaded from there, otherwise
        ``None``. The filename is used by Python for the tracebacks
        if no loader extension is used.

        The last item in the tuple is the `uptodate` function.  If auto
        reloading is enabled it's always called to check if the template
        changed.  No arguments are passed so the function must store the
        old state somewhere (for example in a closure).  If it returns `False`
        the template will be reloaded.
        """
        raise NotImplementedError()

    def list_templates(self) -> t.List[str]:
        """Iterates over all templates.  If the loader does not support that
        it should raise a :exc:`TypeError` which is the default behavior.
        """
        raise TypeError('this loader cannot iterate over all templates')

    @internalcode
    def load(self, environment: 'Environment', name: str, globals: t.Optional[t.MutableMapping[str, t.Any]]=None) -> 'Template':
        """Loads a template.  This method looks up the template in the cache
        or loads one by calling :meth:`get_source`.  Subclasses should not
        override this method as loaders working on collections of other
        loaders (such as :class:`PrefixLoader` or :class:`ChoiceLoader`)
        will not call this method but `get_source` directly.
        """
        source, filename, uptodate = self.get_source(environment, name)
        code = environment.compile(source, name, filename)
        return environment.template_class.from_code(environment, code, globals, uptodate)

class FileSystemLoader(BaseLoader):
    """Load templates from a directory in the file system.

    The path can be relative or absolute. Relative paths are relative to
    the current working directory.

    .. code-block:: python

        loader = FileSystemLoader("templates")

    A list of paths can be given. The directories will be searched in
    order, stopping at the first matching template.

    .. code-block:: python

        loader = FileSystemLoader(["/override/templates", "/default/templates"])

    :param searchpath: A path, or list of paths, to the directory that
        contains the templates.
    :param encoding: Use this encoding to read the text from template
        files.
    :param followlinks: Follow symbolic links in the path.

    .. versionchanged:: 2.8
        Added the ``followlinks`` parameter.
    """

    def __init__(self, searchpath: t.Union[str, 'os.PathLike[str]', t.Sequence[t.Union[str, 'os.PathLike[str]']]], encoding: str='utf-8', followlinks: bool=False) -> None:
        if not isinstance(searchpath, abc.Iterable) or isinstance(searchpath, str):
            searchpath = [searchpath]
        self.searchpath = [os.fspath(p) for p in searchpath]
        self.encoding = encoding
        self.followlinks = followlinks

class PackageLoader(BaseLoader):
    """Load templates from a directory in a Python package.

    :param package_name: Import name of the package that contains the
        template directory.
    :param package_path: Directory within the imported package that
        contains the templates.
    :param encoding: Encoding of template files.

    The following example looks up templates in the ``pages`` directory
    within the ``project.ui`` package.

    .. code-block:: python

        loader = PackageLoader("project.ui", "pages")

    Only packages installed as directories (standard pip behavior) or
    zip/egg files (less common) are supported. The Python API for
    introspecting data in packages is too limited to support other
    installation methods the way this loader requires.

    There is limited support for :pep:`420` namespace packages. The
    template directory is assumed to only be in one namespace
    contributor. Zip files contributing to a namespace are not
    supported.

    .. versionchanged:: 3.0
        No longer uses ``setuptools`` as a dependency.

    .. versionchanged:: 3.0
        Limited PEP 420 namespace package support.
    """

    def __init__(self, package_name: str, package_path: 'str'='templates', encoding: str='utf-8') -> None:
        package_path = os.path.normpath(package_path).rstrip(os.path.sep)
        if package_path == os.path.curdir:
            package_path = ''
        elif package_path[:2] == os.path.curdir + os.path.sep:
            package_path = package_path[2:]
        self.package_path = package_path
        self.package_name = package_name
        self.encoding = encoding
        import_module(package_name)
        spec = importlib.util.find_spec(package_name)
        assert spec is not None, 'An import spec was not found for the package.'
        loader = spec.loader
        assert loader is not None, 'A loader was not found for the package.'
        self._loader = loader
        self._archive = None
        template_root = None
        if isinstance(loader, zipimport.zipimporter):
            self._archive = loader.archive
            pkgdir = next(iter(spec.submodule_search_locations))
            template_root = os.path.join(pkgdir, package_path).rstrip(os.path.sep)
        else:
            roots: t.List[str] = []
            if spec.submodule_search_locations:
                roots.extend(spec.submodule_search_locations)
            elif spec.origin is not None:
                roots.append(os.path.dirname(spec.origin))
            for root in roots:
                root = os.path.join(root, package_path)
                if os.path.isdir(root):
                    template_root = root
                    break
        if template_root is None:
            raise ValueError(f'The {package_name!r} package was not installed in a way that PackageLoader understands.')
        self._template_root = template_root

class DictLoader(BaseLoader):
    """Loads a template from a Python dict mapping template names to
    template source.  This loader is useful for unittesting:

    >>> loader = DictLoader({'index.html': 'source here'})

    Because auto reloading is rarely useful this is disabled per default.
    """

    def __init__(self, mapping: t.Mapping[str, str]) -> None:
        self.mapping = mapping

class FunctionLoader(BaseLoader):
    """A loader that is passed a function which does the loading.  The
    function receives the name of the template and has to return either
    a string with the template source, a tuple in the form ``(source,
    filename, uptodatefunc)`` or `None` if the template does not exist.

    >>> def load_template(name):
    ...     if name == 'index.html':
    ...         return '...'
    ...
    >>> loader = FunctionLoader(load_template)

    The `uptodatefunc` is a function that is called if autoreload is enabled
    and has to return `True` if the template is still up to date.  For more
    details have a look at :meth:`BaseLoader.get_source` which has the same
    return value.
    """

    def __init__(self, load_func: t.Callable[[str], t.Optional[t.Union[str, t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]]]]) -> None:
        self.load_func = load_func

class PrefixLoader(BaseLoader):
    """A loader that is passed a dict of loaders where each loader is bound
    to a prefix.  The prefix is delimited from the template by a slash per
    default, which can be changed by setting the `delimiter` argument to
    something else::

        loader = PrefixLoader({
            'app1':     PackageLoader('mypackage.app1'),
            'app2':     PackageLoader('mypackage.app2')
        })

    By loading ``'app1/index.html'`` the file from the app1 package is loaded,
    by loading ``'app2/index.html'`` the file from the second.
    """

    def __init__(self, mapping: t.Mapping[str, BaseLoader], delimiter: str='/') -> None:
        self.mapping = mapping
        self.delimiter = delimiter

class ChoiceLoader(BaseLoader):
    """This loader works like the `PrefixLoader` just that no prefix is
    specified.  If a template could not be found by one loader the next one
    is tried.

    >>> loader = ChoiceLoader([
    ...     FileSystemLoader('/path/to/user/templates'),
    ...     FileSystemLoader('/path/to/system/templates')
    ... ])

    This is useful if you want to allow users to override builtin templates
    from a different location.
    """

    def __init__(self, loaders: t.Sequence[BaseLoader]) -> None:
        self.loaders = loaders

class _TemplateModule(ModuleType):
    """Like a normal module but with support for weak references"""

class ModuleLoader(BaseLoader):
    """This loader loads templates from precompiled templates.

    Example usage:

    >>> loader = ChoiceLoader([
    ...     ModuleLoader('/path/to/compiled/templates'),
    ...     FileSystemLoader('/path/to/templates')
    ... ])

    Templates can be precompiled with :meth:`Environment.compile_templates`.
    """
    has_source_access = False

    def __init__(self, path: t.Union[str, 'os.PathLike[str]', t.Sequence[t.Union[str, 'os.PathLike[str]']]]) -> None:
        package_name = f'_jinja2_module_templates_{id(self):x}'
        mod = _TemplateModule(package_name)
        if not isinstance(path, abc.Iterable) or isinstance(path, str):
            path = [path]
        mod.__path__ = [os.fspath(p) for p in path]
        sys.modules[package_name] = weakref.proxy(mod, lambda x: sys.modules.pop(package_name, None))
        self.module = mod
        self.package_name = package_name
