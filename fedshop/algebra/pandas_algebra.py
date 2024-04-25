from itertools import chain
from pprint import pprint
from pyparsing import Forward, Group, Literal, Suppress, Word, ZeroOrMore, alphanums, alphas, infixNotation, oneOf, opAssoc, Optional
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.plugins.sparql.algebra import _traverseAgg, traverse

""" 
Expr := Comparison | UnaryExpr | BinaryExpr
BinaryExpr := Expr BinaryOperator Expr
UnaryExpr := UnaryOperator Exprs
UnaryOperator := "not"
BinaryOperator := "and" | "or" | "in" | "not in"
Comparison := Column ColumnOp Column
ColumnOp := "!=" | ">" | "<" | ">=" | "<=" | "=="
Column := `Word`

Example:
Input: "`a` > `b` and (`c` < `d` or `e` == `f`)"
Output:
[
    BinaryExpr_{
        'left': Comparison_{'left': 'a', 'op': '>', 'right': 'b'}, 
        'op': 'and', 
        'right': BinaryExpr_{
            'left': Comparison_{'left': 'c', 'op': '<', 'right': 'd'}, 
            'op': 'or', 
            'right': Comparison_{'left': 'e', 'op': '==', 'right': 'f'}
        }
    }
]

"""

# Define grammar for identifier (variable names)
Column = Suppress("`") + Word(alphas, alphanums + "_") + Suppress("`")
Column.setParseAction(lambda t: CompValue("Column", column_name=t[0]))

AccessVariable = Literal("@") + Word(alphas, alphanums + "_")
AccessVariable.setParseAction(lambda t: CompValue("AccessVariable", var=t[1]))

Term = Column | AccessVariable

# Define condition operators
ColumnBinaryOp = oneOf("!= > < >= <= == in", use_regex=False, as_keyword=True)
ColumnBinaryOp.setParseAction(lambda t: CompValue("ColumnBinaryOp", op=t[0]))

ColumnUnaryOp = oneOf("~", use_regex=False, as_keyword=True)
ColumnUnaryOp.setParseAction(lambda t: CompValue("ColumnUnaryOp", op=t[0]))

ComparisonCondition = Term + ColumnBinaryOp + Term
ComparisonCondition.setParseAction(lambda t: CompValue("ComparisonCondition", left=t[0], op=t[1], right=t[2]))

Function = Term + Suppress(".") + Word(alphas, alphanums + "_") + Suppress("(") + ZeroOrMore(Term) + Suppress(")")
Function.setParseAction(lambda t: CompValue("Function", left=t[0], func=t[1], args=t[2:]))

def FunctionExpr(t):
    t = t[0]
    return CompValue("FunctionCondition", op=t[0], expr=t[1])

def ComposedFunctionExpr(t):
    t = t[0]
    return CompValue("FunctionCondition", outer=t[0], right=t[1])

FunctionCondition = infixNotation(
    Function,
    [
        (ColumnUnaryOp, 1, opAssoc.RIGHT, FunctionExpr),
        (Function, 1, opAssoc.RIGHT, ComposedFunctionExpr)
    ]
)

Condition = ComparisonCondition | FunctionCondition

# Define logical operators
LogicalBinaryOp = oneOf("and or", use_regex=False, as_keyword=True)
LogicalBinaryOp.setParseAction(lambda t: CompValue("LogicalBinaryOp", op=t[0]))

LogicalUnaryOp = oneOf("not", use_regex=False, as_keyword=True) 
LogicalUnaryOp.setParseAction(lambda t: CompValue("LogicalUnaryOp", op=t[0]))

def BinaryExpr(t):
    t = t[0]
    return CompValue("BinaryExpr", left=t[0], op=t[1], right=t[2])

def UnaryExpr(t):
    t = t[0]
    return CompValue("UnaryExpr", op=t[0], right=t[1])

Expr = Forward()
Expr = Group(infixNotation(
    Condition | Expr,
    [
        (LogicalBinaryOp, 2, opAssoc.RIGHT, BinaryExpr),
        (LogicalUnaryOp, 1, opAssoc.RIGHT, UnaryExpr)
    ]
))
#Expr.setParseAction(lambda t: CompValue("Expr", expr=t[0]))

# Utility functions

def collect_constants(node, children):
    if isinstance(node, CompValue):
        if node.name == "Column":
            children.append([node["column_name"]])
    return list(chain(*children))

def translate_query(algebra):
    def _to_string(node):
        if isinstance(node, CompValue):
            if node.name == "Expr":
                return _to_string(node["expr"])
            
            elif node.name == "ComparisonCondition":
                left = _to_string(node["left"])
                op = _to_string(node["op"])
                right = _to_string(node["right"])
                return f"{left} {op} {right}"
            
            elif node.name == "FunctionCondition":
                raise NotImplementedError("FunctionCondition translation not implemented")
            
            elif node.name == "BinaryExpr":
                left = _to_string(node["left"])
                op = _to_string(node["op"])
                right = _to_string(node["right"])
                return f"({left} {op} {right})"
            
            elif node.name == "UnaryExpr":
                op = _to_string(node["op"])
                right = _to_string(node["right"])
                return f"{op} {right}"
            
            elif node.name == "Column":
                return f"`{node['column_name']}`"
            
            elif node.name == "AccessVariable":
                return f"@{node['var']}"
            
            elif node.name in ["LogicalBinaryOp", "LogicalUnaryOp", "ColumnBinaryOp", "ColumnUnaryOp"]:
                return node["op"]
            
            else:
                raise NotImplementedError(f"Translation for {node.name} not implemented")
    
    return list(chain(*traverse(algebra, _to_string)))[0]
                    
def parse_expr(input_expr):
    parsed_expr = Expr.parseString(input_expr)
    return parsed_expr

# input_expr = "`ProductFeature2` != `ProductFeature1` and `x` <= `p1` and `ProductFeature3` != `ProductFeature2` and `ProductFeature3` != `ProductFeature1` and `y` <= `p2`"
# input_expr = "`a` == @b and `c`.isin(`d`.groupby(`localProduct`)) and `e` == `f`"
# input_expr = "`p3` <= `p1` and (`y` <= `p2` or `p3` <= `p2`) and `y` <= `p1`"
# input_expr = "not (`a` > `b`) and (`c` < `d` or `e` == `f`)"
# algebra = parse_expr(input_expr)
# print(algebra)
