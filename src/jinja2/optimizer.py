"""The optimizer tries to constant fold expressions and modify the AST
in place so that it should be faster to evaluate.

Because the AST does not contain all the scoping information and the
compiler has to find that out, we cannot do all the optimizations we
want. For example, loop unrolling doesn't work because unrolled loops
would have a different scope. The solution would be a second syntax tree
that stored the scoping rules.
"""
import typing as t
from . import nodes
from .visitor import NodeTransformer
if t.TYPE_CHECKING:
    from .environment import Environment

def optimize(node: nodes.Node, environment: 'Environment') -> nodes.Node:
    """The context hint can be used to perform a static optimization
    based on the context given."""
    optimizer = Optimizer(environment)
    return optimizer.visit(node)

class Optimizer(NodeTransformer):

    def __init__(self, environment: 't.Optional[Environment]') -> None:
        self.environment = environment

    def visit_Const(self, node: nodes.Const) -> nodes.Node:
        return node

    def visit_List(self, node: nodes.List) -> nodes.Node:
        for idx, item in enumerate(node.items):
            if isinstance(item, nodes.Const):
                continue
            node.items[idx] = self.visit(item)
        return node

    def visit_Dict(self, node: nodes.Dict) -> nodes.Node:
        for idx, pair in enumerate(node.items):
            if isinstance(pair.key, nodes.Const) and isinstance(pair.value, nodes.Const):
                continue
            node.items[idx] = nodes.Pair(self.visit(pair.key), self.visit(pair.value))
        return node

    def visit_Concat(self, node: nodes.Concat) -> nodes.Node:
        # Optimize concatenation of constant strings
        optimized_nodes = []
        current_const = []
        for n in node.nodes:
            if isinstance(n, nodes.Const) and isinstance(n.value, str):
                current_const.append(n.value)
            else:
                if current_const:
                    optimized_nodes.append(nodes.Const(''.join(current_const)))
                    current_const = []
                optimized_nodes.append(self.visit(n))
        if current_const:
            optimized_nodes.append(nodes.Const(''.join(current_const)))
        if len(optimized_nodes) == 1:
            return optimized_nodes[0]
        return nodes.Concat(optimized_nodes)

    def visit_CondExpr(self, node: nodes.CondExpr) -> nodes.Node:
        node.test = self.visit(node.test)
        node.expr1 = self.visit(node.expr1)
        if node.expr2 is not None:
            node.expr2 = self.visit(node.expr2)
        if isinstance(node.test, nodes.Const):
            if node.test.value:
                return node.expr1
            return node.expr2 if node.expr2 is not None else nodes.Const(None)
        return node

    def visit_Filter(self, node: nodes.Filter) -> nodes.Node:
        node.node = self.visit(node.node)
        node.args = [self.visit(arg) for arg in node.args]
        for kwarg in node.kwargs:
            kwarg.value = self.visit(kwarg.value)
        return node

    def visit_Test(self, node: nodes.Test) -> nodes.Node:
        node.node = self.visit(node.node)
        node.args = [self.visit(arg) for arg in node.args]
        for kwarg in node.kwargs:
            kwarg.value = self.visit(kwarg.value)
        return node

    def visit_Call(self, node: nodes.Call) -> nodes.Node:
        node.node = self.visit(node.node)
        node.args = [self.visit(arg) for arg in node.args]
        for kwarg in node.kwargs:
            kwarg.value = self.visit(kwarg.value)
        if node.dyn_args is not None:
            node.dyn_args = self.visit(node.dyn_args)
        if node.dyn_kwargs is not None:
            node.dyn_kwargs = self.visit(node.dyn_kwargs)
        return node

    def generic_visit(self, node: nodes.Node) -> nodes.Node:
        for field, value in node.iter_fields():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, nodes.Node):
                        self.visit(item)
            elif isinstance(value, nodes.Node):
                self.visit(value)
        return node
