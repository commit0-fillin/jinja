"""API for traversing the AST nodes. Implemented by the compiler and
meta introspection.
"""
import typing as t
from .nodes import Node
if t.TYPE_CHECKING:
    import typing_extensions as te

    class VisitCallable(te.Protocol):

        def __call__(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any:
            ...

class NodeVisitor:
    """Walks the abstract syntax tree and call visitor functions for every
    node found.  The visitor functions may return values which will be
    forwarded by the `visit` method.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `get_visitor` function.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.
    """

    def get_visitor(self, node: Node) -> 't.Optional[VisitCallable]':
        """Return the visitor function for this node or `None` if no visitor
        exists for this node.  In that case the generic visit function is
        used instead.
        """
        method = 'visit_' + node.__class__.__name__
        return getattr(self, method, None)

    def visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Visit a node."""
        f = self.get_visitor(node)
        if f is not None:
            return f(node, *args, **kwargs)
        return self.generic_visit(node, *args, **kwargs)

    def generic_visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Called if no explicit visitor function exists for a node."""
        return None

class NodeTransformer(NodeVisitor):
    """Walks the abstract syntax tree and allows modifications of nodes.

    The `NodeTransformer` will walk the AST and use the return value of the
    visitor functions to replace or remove the old node.  If the return
    value of the visitor function is `None` the node will be removed
    from the previous location otherwise it's replaced with the return
    value.  The return value may be the original node in which case no
    replacement takes place.
    """

    def visit_list(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.List[Node]:
        """As transformers may return lists in some places this method
        can be used to enforce a list as return value.
        """
        result = self.visit(node, *args, **kwargs)
        if result is None:
            return []
        elif isinstance(result, list):
            return result
        else:
            return [result]
