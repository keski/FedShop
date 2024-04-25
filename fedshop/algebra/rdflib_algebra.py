import collections
from itertools import chain
from pprint import pprint
import re
from typing import (
    DefaultDict,
    List
)
import typing
import uuid

from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import Query
from rdflib.plugins.sparql.algebra import ExpressionNotCoveredException, traverse, _AlgebraTranslator, _traverseAgg, translateQuery
from rdflib.term import Variable, Literal, URIRef


# ---------------------------
# Some convenience methods
from rdflib.term import Identifier, URIRef, Variable

# Some utility methods

def extract_where(node, children):
    """
    Extracts the 'where' clause from a given node.

    Args:
        node (CompValue): The node to extract the 'where' clause from.
        children (list): The list of children nodes.

    Returns:
        The 'where' clause of the given node.

    """
    if isinstance(node, CompValue):
        if node.name == "SelectQuery":
            return node["where"]
        
    if len(children) > 0:
        return children[-1] # Get the right most child

def collect_variables(node, children):
    """
    Collects variables from the given node and its children.

    Args:
        node (Variable or Node): The node to collect variables from.
        children (list): A list of children nodes.

    Returns:
        list: A list of collected variables.
    """
    
    if isinstance(node, CompValue):
        if node.name == "TriplesBlock":
            for triple in node["triples"]:
                for component in triple:
                    if isinstance(component, Variable):
                        children.append([component])
                    elif isinstance(component, CompValue) and component.name == "vars":
                        children.append([component["var"]])
    return list(chain(*children))
        
def disable_orderby_limit(node):
    """Disable the 'orderby' and 'limitoffset' properties in the given node.

    This function removes the 'orderby' and 'limitoffset' properties from the given node,
    if it is an instance of CompValue.

    Args:
        node (CompValue): The node to disable 'orderby' and 'limitoffset' properties for.

    Returns:
        CompValue: The modified node with 'orderby' and 'limitoffset' properties removed.
    """
    if isinstance(node, CompValue):
        node.pop("orderby", None)
        node.pop("limitoffset", None)
        return node

def inject_constant_into_placeholders(node, injection_dict):
    """
    Recursively inject constant values into placeholders in the query.

    Args:
        node: The current node in the query AST.
        injection_dict: A dictionary mapping variable names to constant values.

    Returns:
        The modified node with constant values injected.
    """
    if isinstance(node, Variable):
        var_name = str(node)
        if var_name in injection_dict:
            injection_value = injection_dict[var_name]
            if isinstance(injection_value, str) and injection_value.startswith("http"):
                return URIRef(injection_value)
            return Literal(injection_dict[var_name])
    elif isinstance(node, CompValue):
        if node.name == "vars":
            var_name = str(node["var"])
            if var_name in injection_dict:
                injection_value = injection_dict[var_name]
                if isinstance(injection_value, str) and injection_value.startswith("http"):
                    return URIRef(injection_value)
                return Literal(injection_dict[var_name])

def add_graph_to_triple_pattern(node):
    """
    Wrap a triple pattern with GRAPH clause

    Args:
        node (CompValue): The triple pattern node to add the graph to.

    Returns:
        CompValue: The modified triple pattern node with the graph added.
    """
    
    if isinstance(node, CompValue):
        if node.name == "TriplesBlock":
            graph_triples = []
            for triple in node["triples"]:
                graph_id = str(uuid.uuid4())[:8]
                graph_node = CompValue(
                    "GraphGraphPattern", 
                    term=Variable(f"g{graph_id}"),
                    graph=CompValue(
                        "GroupGraphPatternSub",
                        part=[CompValue("TriplesBlock", triples=[triple])]
                    )
                )
                graph_triples.append(graph_node)
            
            node = graph_triples
            return node
        elif node.name == "GroupGraphPatternSub":
            new_part = []
            for part in node["part"]:
                if isinstance(part, list):
                    new_part.extend(part)
                else:
                    new_part.append(part)
            node["part"] = new_part
            return node

def collect_graphs_variables(node, children):
    if isinstance(node, CompValue):
        if node.name == "GraphGraphPattern":
            children.append([node["term"]])

    return list(chain(*children))

def replace_select_projection_with_graph(node):
    if isinstance(node, CompValue):
        if node.name in ["SelectQuery", "ConstructQuery", "DescribeQuery", "AskQuery"]:
            graph_vars = _traverseAgg(node["where"], collect_graphs_variables)
            return CompValue(
                "SelectQuery",
                modifier="DISTINCT",
                projection=graph_vars,
                where=node["where"]
            )
        
    return node
    
def remove_expression_with_placeholder(node, consts): 
    
    if len(consts) == 0:
        return node
    
    if isinstance(node, Variable):
        if str(node) not in consts:
            return CompValue("Placeholder")   
        return node
    elif isinstance(node, CompValue): 
        
        is_binary_expr = "expr" in node.keys() and "other" in node.keys()  
        is_unary_expr = "expr" in node.keys() and "other" not in node.keys()         
        
        # If the expression is a placeholder, return a placeholder
        if is_unary_expr:
            expr = node["expr"]
            new_expr = traverse(expr, visitPost=lambda x: remove_expression_with_placeholder(x, consts))
            #print(f"Expression: {new_expr}")
            if isinstance(new_expr, CompValue) and new_expr.name == "Placeholder":
                return CompValue("Placeholder")
            else:
                node["expr"] = new_expr
        
        # Recursively transform the other expression(s)
        elif is_binary_expr:
            has_empty_expr = False
            has_empty_other = False
            
            expr = node["expr"]
            new_expr = traverse(expr, visitPost=lambda x: remove_expression_with_placeholder(x, consts))
            #logger.debug(f"Expression: {new_expr}")
            if isinstance(new_expr, CompValue) and new_expr.name == "Placeholder":
                has_empty_expr = True
            else:
                node["expr"] = new_expr
                
            other = node["other"]
            # If the other expression is a placeholder, return a placeholder
            if isinstance(other, CompValue) and other.name == "Placeholder":
                return CompValue("Placeholder")
                    
            # If the other expression is a list, pop the first element and transform it
            if not isinstance(other, list):
                other = [other]
        
            new_other = []
            for o in other:
                new_o = traverse(o, visitPost=lambda x: remove_expression_with_placeholder(x, consts))
                if isinstance(new_o, CompValue) and new_o.name == "Placeholder":
                    continue
                new_other.append(new_o)
                
            if len(new_other) == 1:
                new_other = new_other[0]

            if len(new_other) == 0:
                has_empty_other = True
                                    
            if not has_empty_expr and not has_empty_other:
                node["expr"] = new_expr
                node["other"] = new_other
            elif has_empty_expr and has_empty_other:
                return CompValue("Placeholder")
            if has_empty_expr:
                node["expr"] = new_other.pop(0)
                
                if len(new_other) == 0:
                    has_empty_other = True
                elif len(new_other) == 1:
                    node["other"] = new_other[0]
                else:
                    node["other"] = new_other
        
            if has_empty_other:
                node.pop("other", None)
        elif node.name == "Function":
            # Should have been treated in the previous step
            raise NotImplementedError("Function expressions are not supported yet")
        elif node.name.startswith("Builtin"):
            if node.name == "Builtin_BOUND":
                arg = node["arg"]
                new_arg = traverse(arg, visitPost=lambda x: remove_expression_with_placeholder(x, consts))
                if isinstance(new_arg, CompValue) and new_arg.name == "Placeholder":
                    return CompValue("Placeholder")
                node["arg"] = new_arg
            elif node.name == "Builtin_REGEX":
                # Should have been treated in the previous 
                new_text = traverse(node["text"], visitPost=lambda x: remove_expression_with_placeholder(x, consts))
                if isinstance(new_text, CompValue) and new_text.name == "Placeholder":
                    return CompValue("Placeholder")
                
                new_pattern = traverse(node["pattern"], visitPost=lambda x: remove_expression_with_placeholder(x, consts))
                if isinstance(new_pattern, CompValue) and new_pattern.name == "Placeholder":
                    return CompValue("Placeholder")
                
                node["text"] = new_text
                node["pattern"] = new_pattern          
                
        return node  

def remove_filter_with_placeholders(node, consts):   
    query_consts = set(consts["query"])
    
    select_consts = None
    if "select" in consts.keys():
        select_consts = set(consts["select"]) & query_consts
    filter_consts = set(consts["filter"]) & query_consts
            
    if isinstance(node, CompValue):
        if node.name == "SelectQuery":
            if select_consts:
                node["projection"] = list(map(lambda x: CompValue("vars", var=Variable(x)), select_consts))
            node["modifier"] = "DISTINCT"
            return node
        
        if node.name == "Filter":
            return traverse(node, visitPost=lambda x: remove_expression_with_placeholder(x, filter_consts))
        
        if "part" in node.keys():
            if isinstance(node["part"], list):
                new_parts = []
                for part in node["part"]:
                    if isinstance(part, CompValue) and part.name == "Placeholder":
                        continue
                    new_parts.append(part)
                    
                node["part"] = new_parts
                
        node.pop("orderby", None)
        node.pop("limitoffset", None)
        return node
    
def add_values_with_placeholders(node, inline_data):
    if isinstance(node, CompValue):
        if node.name == "SelectQuery":
            inline_data_keys = [ Variable(k) for k in inline_data.keys() ] 
            inline_data_values = [ 
                [ URIRef(v) if v.startswith("http") else Literal(v) for v in values ]
                for values in inline_data.values()
            ]
            if len(inline_data_keys) == 1:
                inline_data_values = inline_data_values[0]
                
            values_clause = CompValue(
                "InlineData",
                var = inline_data_keys,
                value = inline_data_values
            )
            
            node["where"]["part"].insert(0, values_clause)
            return node
        
def add_service_to_triple_blocks(node, inline_data):
    if isinstance(node, "CompValue"):
        for node_attr, node_value in node.items():
            if node_attr in ["where"]: continue
            if isinstance(node_value, CompValue) and node_value.name == "GroupGraphPatternSub":
                service_node = CompValue(
                    "ServiceGraphPattern",
                    service_string = translateAlgebra(translateQuery(node)),
                    term = Variable(graph_id = str(uuid.uuid4())[:8]),
                    graph = node
                )
                node[node_attr] = service_node
        return node                            

# --------- MISC ------------
class PrettyAlgebraTranslator(_AlgebraTranslator):
    """
    Translator of a Query's algebra to its equivalent SPARQL (string).

    Coded as a class to support storage of state during the translation process,
    without use of a file.

    Anticipated Usage:

    .. code-block:: python

        translated_query = _AlgebraTranslator(query).translateAlgebra()

    An external convenience function which wraps the above call,
    `translateAlgebra`, is supplied, so this class does not need to be
    referenced by client code at all in normal use.
    """

    def __init__(self, query_algebra: Query, **kwargs):
        self.query_algebra = query_algebra
        self.aggr_vars: DefaultDict[
            Identifier, List[Identifier]
        ] = collections.defaultdict(list)
        self._alg_translation: str = ""
        self.__identation_level = 0
        self.__identation_token = " " * int(kwargs.get("identation_n_space", 4))
        self.__breakline_token = "\n"
        
    def _replace(
        self,
        old: str,
        new: str,
        search_from_match: str = None,
        search_from_match_occurrence: int = None,
        count: int = 1,
    ):  
        def find_nth(haystack, needle, n):
            start = haystack.lower().find(needle)
            while start >= 0 and n > 1:
                start = haystack.lower().find(needle, start + len(needle))
                n -= 1
            return start

        if search_from_match and search_from_match_occurrence:
            position = find_nth(
                self._alg_translation, search_from_match, search_from_match_occurrence
            )
            filedata_pre = self._alg_translation[:position]
            filedata_post = (
                self._alg_translation[position:].replace(old, new, count)
            )
            self._alg_translation = filedata_pre + filedata_post
        else:
            self._alg_translation = (
                self._alg_translation.replace(old, new, count)
            )        

    def convert_node_arg(
        self, node_arg: typing.Union[Identifier, CompValue, Expr, str]
    ) -> str:
        if isinstance(node_arg, Identifier):
            if node_arg in self.aggr_vars.keys():
                # type error: "Identifier" has no attribute "n3"
                grp_var = self.aggr_vars[node_arg].pop(0).n3()  # type: ignore[attr-defined]
                return grp_var
            else:
                # type error: "Identifier" has no attribute "n3"
                return node_arg.n3()  # type: ignore[attr-defined]
        elif isinstance(node_arg, CompValue):
            return "{" + node_arg.name + "}"
        elif isinstance(node_arg, Expr):
            return "{" + node_arg.name + "}"
        elif isinstance(node_arg, str):
            return node_arg
        else:
            raise ExpressionNotCoveredException(
                "The expression {0} might not be covered yet.".format(node_arg)
            )

    def sparql_query_text(self, node):
        """
         https://www.w3.org/TR/sparql11-query/#sparqlSyntax

        :param node:
        :return:
        """
        
        if isinstance(node, CompValue):
            identation = self.__identation_token * self.__identation_level 
            next_level_identation = self.__identation_token * (self.__identation_level + 1)
                
            # 18.2 Query Forms
            if node.name == "SelectQuery":
                self._alg_translation = (
                    identation + "-*-SELECT-*- " + "{" + node.p.name + "}" + self.__breakline_token +
                    identation + self.__breakline_token
                )
                self.__identation_level += 1
                
            # 18.2 Graph Patterns
            elif node.name == "BGP":
                # Identifiers or Paths
                # Negated path throws a type error. Probably n3() method of negated paths should be fixed
                triples = "".join(
                    identation + triple[0].n3() + " " + triple[1].n3() + " " + triple[2].n3() + "." + self.__breakline_token
                    for triple in node.triples
                )
                self._replace("{BGP}", triples)
                # The dummy -*-SELECT-*- is placed during a SelectQuery or Multiset pattern in order to be able
                # to match extended variables in a specific Select-clause (see "Extend" below)
                self._replace("-*-SELECT-*-", "SELECT", count=-1)
                # If there is no "Group By" clause the placeholder will simply be deleted. Otherwise there will be
                # no matching {GroupBy} placeholder because it has already been replaced by "group by variables"
                self._replace("{GroupBy}", "", count=-1)
                self._replace("{Having}", "", count=-1)
            elif node.name == "Join":
                self._replace(
                    "{Join}", (
                        #identation + "{" + self.__breakline_token + 
                        "{" + node.p1.name + "}" + self.__breakline_token + 
                        #identation + "}" + "." + self.__breakline_token + 

                        #identation + "{" + self.__breakline_token + 
                        "{" + node.p2.name + "}" + self.__breakline_token 
                        #identation + "}" + self.__breakline_token
                    )
                )
                #self.__identation_level += 1
            elif node.name == "LeftJoin":
                self._replace(
                    "{LeftJoin}",
                    (
                        identation + "{" + node.p1.name + "}" + self.__breakline_token + 
                        identation + "OPTIONAL" + "{" + self.__breakline_token + 
                        identation + "{" + node.p2.name + "}" + self.__breakline_token +
                        identation + "}" + self.__breakline_token
                    ),
                )
                self.__identation_level += 1
            elif node.name == "Filter":
                if isinstance(node.expr, CompValue):
                    expr = node.expr.name
                else:
                    raise ExpressionNotCoveredException(
                        "This expression might not be covered yet."
                    )
                if node.p:
                    # Filter with p=AggregateJoin = Having
                    if node.p.name == "AggregateJoin":
                        self._replace(
                            "{Filter}", 
                            (
                                identation + self.__breakline_token + 
                                identation + "{" + node.p.name + "}" + self.__breakline_token
                            )
                        )
                        self._replace(
                            "{Having}", 
                            (
                                identation + "HAVING({" + expr + "})" + self.__breakline_token
                            )
                        )
                    else:
                        self._replace(
                            "{Filter}", 
                            (
                                identation + "FILTER({" + expr + "})" + self.__breakline_token + 
                                "{" + node.p.name +  "}" + self.__breakline_token
                            )
                        )
                else:
                    self._replace(
                        "{Filter}", 
                        identation + "FILTER({" + expr + "})" + self.__breakline_token
                    )

            elif node.name == "Union":
                self._replace(
                    "{Union}", (
                        identation + "{" + self.__breakline_token +
                        next_level_identation + "{" + node.p1.name + "}" + self.__breakline_token + 
                        identation + "} UNION {" + self.__breakline_token +
                        next_level_identation + "{" + node.p2.name + "}" + self.__breakline_token + 
                        identation + "}" + self.__breakline_token
                    )
                )
            elif node.name == "Graph":
                expr = (
                    identation + "GRAPH " + node.term.n3() + " {" + self.__breakline_token + 
                    next_level_identation + "{" + node.p.name + "}" + self.__breakline_token + 
                    identation + "}" + self.__breakline_token
                )
                self._replace("{Graph}", expr)
            elif node.name == "Extend":
                query_string = self._alg_translation.lower()
                select_occurrences = query_string.count("-*-select-*-")
                self._replace(
                    node.var.n3(),
                    "("
                    + self.convert_node_arg(node.expr)
                    + " as "
                    + node.var.n3()
                    + ")",
                    search_from_match="-*-select-*-",
                    search_from_match_occurrence=select_occurrences,
                )
                self._replace(
                    "{Extend}", 
                    (
                        identation + "{" + node.p.name + "}"
                    )
                )
            elif node.name == "Minus":
                expr = (
                    next_level_identation + "{" + node.p1.name + "}" + self.__breakline_token +
                    identation + "} MINUS {" + self.__breakline_token + 
                    next_level_identation + "{" + node.p2.name + "}" + self.__breakline_token +
                    identation + "}" + self.__breakline_token
                )
                self._replace("{Minus}", expr)
            elif node.name == "Group":
                group_by_vars = []
                if node.expr:
                    for var in node.expr:
                        if isinstance(var, Identifier):
                            group_by_vars.append(var.n3())
                        else:
                            raise ExpressionNotCoveredException(
                                "This expression might not be covered yet."
                            )
                    self._replace(
                        "{Group}", 
                        (
                            "{" + node.p.name + "}" 
                        )
                    )
                    self._replace(
                        "{GroupBy}", "GROUP BY " + " ".join(group_by_vars) + " "
                    )
                else:
                    self._replace(
                        "{Group}", 
                        (
                            "{" + node.p.name + "}" 
                        )
                    )
            elif node.name == "AggregateJoin":
                self._replace(
                    "{AggregateJoin}", 
                    (
                        "{" + node.p.name + "}" 
                    )
                )
                for agg_func in node.A:
                    if isinstance(agg_func.res, Identifier):
                        identifier = agg_func.res.n3()
                    else:
                        raise ExpressionNotCoveredException(
                            "This expression might not be covered yet."
                        )
                    self.aggr_vars[agg_func.res].append(agg_func.vars)

                    agg_func_name = agg_func.name.split("_")[1]
                    distinct = ""
                    if agg_func.distinct:
                        distinct = agg_func.distinct + " "
                    if agg_func_name == "GroupConcat":
                        self._replace(
                            identifier,
                            "GROUP_CONCAT"
                            + "("
                            + distinct
                            + agg_func.vars.n3()
                            + ";SEPARATOR="
                            + agg_func.separator.n3()
                            + ")",
                        )
                    else:
                        self._replace(
                            identifier,
                            agg_func_name.upper()
                            + "("
                            + distinct
                            + self.convert_node_arg(agg_func.vars)
                            + ")",
                        )
                    # For non-aggregated variables the aggregation function "sample" is automatically assigned.
                    # However, we do not want to have "sample" wrapped around non-aggregated variables. That is
                    # why we replace it. If "sample" is used on purpose it will not be replaced as the alias
                    # must be different from the variable in this case.
                    self._replace(
                        "(SAMPLE({0}) as {0})".format(
                            self.convert_node_arg(agg_func.vars)
                        ),
                        self.convert_node_arg(agg_func.vars),
                    )
            elif node.name == "GroupGraphPatternSub":
                self._replace(
                    "GroupGraphPatternSub",
                    (
                        identation + "{" + self.__breakline_token +
                        "".join([
                            next_level_identation + self.convert_node_arg(pattern) 
                            for pattern in node.part
                        ]) +
                        identation + "}" + self.__breakline_token
                    ),
                )
                self.__identation_level += 1
                
            elif node.name == "TriplesBlock":
                print("triplesblock")
                self._replace(
                    "{TriplesBlock}",
                    (
                        #identation + "{" + self.__breakline_token +
                        "".join(
                            identation
                            + triple[0].n3()
                            + " "
                            + triple[1].n3()
                            + " "
                            + triple[2].n3()
                            + "."
                            + self.__breakline_token
                            for triple in node.triples
                        )
                        #identation + "}" + self.__breakline_token
                    ),
                )

            # 18.2 Solution modifiers
            elif node.name == "ToList":
                raise ExpressionNotCoveredException(
                    "This expression might not be covered yet."
                )
            elif node.name == "OrderBy":
                order_conditions = []
                for c in node.expr:
                    if isinstance(c.expr, Identifier):
                        var = c.expr.n3()
                        if c.order is not None:
                            cond = c.order + "(" + var + ")"
                        else:
                            cond = var
                        order_conditions.append(cond)
                    else:
                        raise ExpressionNotCoveredException(
                            "This expression might not be covered yet."
                        )
                self._replace("{OrderBy}", "{" + node.p.name + "}")
                self._replace("{OrderConditions}", " ".join(order_conditions) + " ")
            elif node.name == "Project":
                project_variables = []
                for var in node.PV:
                    if isinstance(var, Identifier):
                        project_variables.append(var.n3())
                    else:
                        raise ExpressionNotCoveredException(
                            "This expression might not be covered yet."
                        )
                order_by_pattern = ""
                if node.p.name == "OrderBy":
                    order_by_pattern = "ORDER BY {OrderConditions}"
                self._replace(
                    "{Project}",
                    " ".join(project_variables) + " "
                    + "{" + self.__breakline_token
                    + "{" + node.p.name + "}" + self.__breakline_token
                    + "}" + self.__breakline_token
                    + "{GroupBy}" + self.__breakline_token
                    + order_by_pattern + self.__breakline_token
                    + "{Having}" + self.__breakline_token,
                )
            elif node.name == "Distinct":
                self._replace("{Distinct}", "DISTINCT {" + node.p.name + "}")
            elif node.name == "Reduced":
                self._replace("{Reduced}", "REDUCED {" + node.p.name + "}")
            elif node.name == "Slice":
                slice = "OFFSET " + str(node.start) + " LIMIT " + str(node.length)
                self._replace(
                    "{Slice}", 
                    "{" + node.p.name + "}" + slice
                )
            elif node.name == "ToMultiSet":
                if node.p.name == "values":
                    self._replace(
                        "{ToMultiSet}", 
                        (
                            #identation + "{" + self.__breakline_token +
                            next_level_identation + "{" + node.p.name + "}" + self.__breakline_token 
                            #identation + "}" + self.__breakline_token
                        )
                    )
                else:
                    self._replace(
                        "{ToMultiSet}", 
                        (
                            #identation + "{" + self.__breakline_token + 
                            next_level_identation + "-*-SELECT-*- " + "{" + node.p.name + "}" + self.__breakline_token
                            #identation + "}" + self.__breakline_token
                        )
                    )

            # 18.2 Property Path

            # 17 Expressions and Testing Values
            # # 17.3 Operator Mapping
            elif node.name == "RelationalExpression":
                expr = self.convert_node_arg(node.expr)
                op = node.op
                if isinstance(list, type(node.other)):
                    other = (
                        "("
                        + ", ".join(self.convert_node_arg(expr) for expr in node.other)
                        + ")"
                    )
                else:
                    other = self.convert_node_arg(node.other)
                condition = "{left} {operator} {right}".format(
                    left=expr, operator=op, right=other
                )
                self._replace("{RelationalExpression}", condition)
            elif node.name == "ConditionalAndExpression":
                inner_nodes = " && ".join(
                    [self.convert_node_arg(expr) for expr in node.other]
                )
                self._replace(
                    "{ConditionalAndExpression}",
                    self.convert_node_arg(node.expr) + " && " + inner_nodes,
                )
            elif node.name == "ConditionalOrExpression":
                inner_nodes = " || ".join(
                    [self.convert_node_arg(expr) for expr in node.other]
                )
                self._replace(
                    "{ConditionalOrExpression}",
                    "(" + self.convert_node_arg(node.expr) + " || " + inner_nodes + ")",
                )
            elif node.name == "MultiplicativeExpression":
                left_side = self.convert_node_arg(node.expr)
                multiplication = left_side
                for i, operator in enumerate(node.op):  # noqa: F402
                    multiplication += (
                        " " + operator + " " + self.convert_node_arg(node.other[i]) + " "
                    )
                self._replace("{MultiplicativeExpression}", multiplication)
            elif node.name == "AdditiveExpression":
                left_side = self.convert_node_arg(node.expr)
                addition = left_side
                for i, operator in enumerate(node.op):
                    addition += (
                        " " + operator + " " + self.convert_node_arg(node.other[i]) + " "
                    )
                self._replace("{AdditiveExpression}", addition)
            elif node.name == "UnaryNot":
                self._replace("{UnaryNot}", "!" + self.convert_node_arg(node.expr))

            # # 17.4 Function Definitions
            # # # 17.4.1 Functional Forms
            elif node.name.endswith("BOUND"):
                bound_var = self.convert_node_arg(node.arg)
                self._replace("{Builtin_BOUND}", "bound(" + bound_var + ")")
            elif node.name.endswith("IF"):
                arg2 = self.convert_node_arg(node.arg2)
                arg3 = self.convert_node_arg(node.arg3)

                if_expression = (
                    "IF(" + "{" + node.arg1.name + "}, " + arg2 + ", " + arg3 + ")"
                )
                self._replace("{Builtin_IF}", if_expression)
            elif node.name.endswith("COALESCE"):
                self._replace(
                    "{Builtin_COALESCE}",
                    "COALESCE("
                    + ", ".join(self.convert_node_arg(arg) for arg in node.arg)
                    + ")",
                )
            elif node.name.endswith("Builtin_EXISTS"):
                # The node's name which we get with node.graph.name returns "Join" instead of GroupGraphPatternSub
                # According to https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#rExistsFunc
                # ExistsFunc can only have a GroupGraphPattern as parameter. However, when we print the query algebra
                # we get a GroupGraphPatternSub
                self._replace(
                    "{Builtin_EXISTS}", 
                    (
                        identation + "EXISTS " + "{{" + self.__breakline_token +
                        next_level_identation + node.graph.name + self.__breakline_token + 
                        identation + "}}" + self.__breakline_token
                    )
                )
                traverse(node.graph, visitPre=self.sparql_query_text)
                return node.graph
            elif node.name.endswith("Builtin_NOTEXISTS"):
                # The node's name which we get with node.graph.name returns "Join" instead of GroupGraphPatternSub
                # According to https://www.w3.org/TR/2013/REC-sparql11-query-20130321/#rNotExistsFunc
                # NotExistsFunc can only have a GroupGraphPattern as parameter. However, when we print the query algebra
                # we get a GroupGraphPatternSub
                print(node.graph.name)
                self._replace(
                    "{Builtin_NOTEXISTS}", 
                    (
                        identation + "NOT EXISTS " + "{{" + self.__breakline_token +
                        next_level_identation + node.graph.name + self.__breakline_token + 
                        identation + "}}" + self.__breakline_token
                    )
                )
                traverse(node.graph, visitPre=self.sparql_query_text)
                return node.graph
            # # # # 17.4.1.5 logical-or: Covered in "RelationalExpression"
            # # # # 17.4.1.6 logical-and: Covered in "RelationalExpression"
            # # # # 17.4.1.7 RDFterm-equal: Covered in "RelationalExpression"
            elif node.name.endswith("sameTerm"):
                self._replace(
                    "{Builtin_sameTerm}",
                    "SAMETERM("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            # # # # IN: Covered in "RelationalExpression"
            # # # # NOT IN: Covered in "RelationalExpression"

            # # # 17.4.2 Functions on RDF Terms
            elif node.name.endswith("Builtin_isIRI"):
                self._replace(
                    "{Builtin_isIRI}", "isIRI(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name.endswith("Builtin_isBLANK"):
                self._replace(
                    "{Builtin_isBLANK}",
                    "isBLANK(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name.endswith("Builtin_isLITERAL"):
                self._replace(
                    "{Builtin_isLITERAL}",
                    "isLITERAL(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name.endswith("Builtin_isNUMERIC"):
                self._replace(
                    "{Builtin_isNUMERIC}",
                    "isNUMERIC(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name.endswith("Builtin_STR"):
                self._replace(
                    "{Builtin_STR}", "STR(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name.endswith("Builtin_LANG"):
                self._replace(
                    "{Builtin_LANG}", "LANG(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name.endswith("Builtin_DATATYPE"):
                self._replace(
                    "{Builtin_DATATYPE}",
                    "DATATYPE(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name.endswith("Builtin_IRI"):
                self._replace(
                    "{Builtin_IRI}", "IRI(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name.endswith("Builtin_BNODE"):
                self._replace(
                    "{Builtin_BNODE}", "BNODE(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name.endswith("STRDT"):
                self._replace(
                    "{Builtin_STRDT}",
                    "STRDT("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("Builtin_STRLANG"):
                self._replace(
                    "{Builtin_STRLANG}",
                    "STRLANG("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("Builtin_UUID"):
                self._replace("{Builtin_UUID}", "UUID()")
            elif node.name.endswith("Builtin_STRUUID"):
                self._replace("{Builtin_STRUUID}", "STRUUID()")

            # # # 17.4.3 Functions on Strings
            elif node.name.endswith("Builtin_STRLEN"):
                self._replace(
                    "{Builtin_STRLEN}",
                    "STRLEN(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name.endswith("Builtin_SUBSTR"):
                args = [self.convert_node_arg(node.arg), node.start]
                if node.length:
                    args.append(node.length)
                expr = "SUBSTR(" + ", ".join(args) + ")"
                self._replace("{Builtin_SUBSTR}", expr)
            elif node.name.endswith("Builtin_UCASE"):
                self._replace(
                    "{Builtin_UCASE}", "UCASE(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name.endswith("Builtin_LCASE"):
                self._replace(
                    "{Builtin_LCASE}", "LCASE(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name.endswith("Builtin_STRSTARTS"):
                self._replace(
                    "{Builtin_STRSTARTS}",
                    "STRSTARTS("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("Builtin_STRENDS"):
                self._replace(
                    "{Builtin_STRENDS}",
                    "STRENDS("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("Builtin_CONTAINS"):
                self._replace(
                    "{Builtin_CONTAINS}",
                    "CONTAINS("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("Builtin_STRBEFORE"):
                self._replace(
                    "{Builtin_STRBEFORE}",
                    "STRBEFORE("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("Builtin_STRAFTER"):
                self._replace(
                    "{Builtin_STRAFTER}",
                    "STRAFTER("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("Builtin_ENCODE_FOR_URI"):
                self._replace(
                    "{Builtin_ENCODE_FOR_URI}",
                    "ENCODE_FOR_URI(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name.endswith("Builtin_CONCAT"):
                expr = "CONCAT({vars})".format(
                    vars=", ".join(self.convert_node_arg(elem) for elem in node.arg)
                )
                self._replace("{Builtin_CONCAT}", expr)
            elif node.name.endswith("Builtin_LANGMATCHES"):
                self._replace(
                    "{Builtin_LANGMATCHES}",
                    "LANGMATCHES("
                    + self.convert_node_arg(node.arg1)
                    + ", "
                    + self.convert_node_arg(node.arg2)
                    + ")",
                )
            elif node.name.endswith("REGEX"):
                args = [
                    self.convert_node_arg(node.text),
                    self.convert_node_arg(node.pattern),
                ]
                expr = "REGEX(" + ", ".join(args) + ")"
                self._replace("{Builtin_REGEX}", expr)
            elif node.name.endswith("REPLACE"):
                self._replace(
                    "{Builtin_REPLACE}",
                    "REPLACE("
                    + self.convert_node_arg(node.arg)
                    + ", "
                    + self.convert_node_arg(node.pattern)
                    + ", "
                    + self.convert_node_arg(node.replacement)
                    + ")",
                )

            # # # 17.4.4 Functions on Numerics
            elif node.name == "Builtin_ABS":
                self._replace(
                    "{Builtin_ABS}", "ABS(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_ROUND":
                self._replace(
                    "{Builtin_ROUND}", "ROUND(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_CEIL":
                self._replace(
                    "{Builtin_CEIL}", "CEIL(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_FLOOR":
                self._replace(
                    "{Builtin_FLOOR}", "FLOOR(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_RAND":
                self._replace("{Builtin_RAND}", "RAND()")

            # # # 17.4.5 Functions on Dates and Times
            elif node.name == "Builtin_NOW":
                self._replace("{Builtin_NOW}", "NOW()")
            elif node.name == "Builtin_YEAR":
                self._replace(
                    "{Builtin_YEAR}", "YEAR(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_MONTH":
                self._replace(
                    "{Builtin_MONTH}", "MONTH(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_DAY":
                self._replace(
                    "{Builtin_DAY}", "DAY(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_HOURS":
                self._replace(
                    "{Builtin_HOURS}", "HOURS(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_MINUTES":
                self._replace(
                    "{Builtin_MINUTES}",
                    "MINUTES(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name == "Builtin_SECONDS":
                self._replace(
                    "{Builtin_SECONDS}",
                    "SECONDS(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name == "Builtin_TIMEZONE":
                self._replace(
                    "{Builtin_TIMEZONE}",
                    "TIMEZONE(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name == "Builtin_TZ":
                self._replace(
                    "{Builtin_TZ}", "TZ(" + self.convert_node_arg(node.arg) + ")"
                )

            # # # 17.4.6 Hash functions
            elif node.name == "Builtin_MD5":
                self._replace(
                    "{Builtin_MD5}", "MD5(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_SHA1":
                self._replace(
                    "{Builtin_SHA1}", "SHA1(" + self.convert_node_arg(node.arg) + ")"
                )
            elif node.name == "Builtin_SHA256":
                self._replace(
                    "{Builtin_SHA256}",
                    "SHA256(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name == "Builtin_SHA384":
                self._replace(
                    "{Builtin_SHA384}",
                    "SHA384(" + self.convert_node_arg(node.arg) + ")",
                )
            elif node.name == "Builtin_SHA512":
                self._replace(
                    "{Builtin_SHA512}",
                    "SHA512(" + self.convert_node_arg(node.arg) + ")",
                )

            # Other
            elif node.name == "values":
                columns = []
                for key in node.res[0].keys():
                    if isinstance(key, Identifier):
                        columns.append(key.n3())
                    else:
                        raise ExpressionNotCoveredException(
                            "The expression {0} might not be covered yet.".format(key)
                        )
                values = "VALUES (" + " ".join(columns) + ")"

                rows = ""
                for elem in node.res:
                    row = []
                    for term in elem.values():
                        if isinstance(term, Identifier):
                            row.append(
                                term.n3()
                            )  # n3() is not part of Identifier class but every subclass has it
                        elif isinstance(term, str):
                            row.append(term)
                        else:
                            raise ExpressionNotCoveredException(
                                "The expression {0} might not be covered yet.".format(
                                    term
                                )
                            )
                    rows += identation + "(" + " ".join(row) + ")" + self.__breakline_token

                self._replace(
                    "{values}", 
                    (
                        identation + values + "{" + self.__breakline_token + 
                        next_level_identation + rows + self.__breakline_token + 
                        identation + "}" + self.__breakline_token 
                    )
                )
            elif node.name == "ServiceGraphPattern":
                self._replace(
                    "{ServiceGraphPattern}",
                    (
                        identation + "SERVICE " + self.convert_node_arg(node.term) + self.__breakline_token +
                        "{" + node.graph.name + "}"
                    )
                )
                traverse(node.graph, visitPre=self.sparql_query_text)
                return node.graph
            
            # else:
            #     raise ExpressionNotCoveredException("The expression {0} might not be covered yet.".format(node.name))

    def translateAlgebra(self) -> str:
        traverse(self.query_algebra.algebra, visitPre=self.sparql_query_text)
        return self._alg_translation
    
def translateAlgebra(query_algebra: Query, pretty=False) -> str:
    """
    Translates a SPARQL 1.1 algebra tree into the corresponding query string.

    :param query_algebra: An algebra returned by `translateQuery`.
    :return: The query form generated from the SPARQL 1.1 algebra tree for
        SELECT queries.
    """
    if pretty:
        return PrettyAlgebraTranslator(query_algebra=query_algebra).translateAlgebra()
    return _AlgebraTranslator(query_algebra=query_algebra).translateAlgebra()