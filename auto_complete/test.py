
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

# ======== Core type vars ========
M_co = TypeVar("M_co", bound="Model", covariant=True)
M = TypeVar("M", bound="Model")
T = TypeVar("T")


# ======== Expression objects ========
@dataclass(frozen=True)
class Expr:
    op: str                 # "exact", "in", "isnull", "icontains", "gt", ...
    path: Tuple[str, ...]   # ("team", "league", "name")
    value: Any = None       # the RHS

    def __and__(self, other: "Expr") -> "Expr":
        return Expr("AND", tuple(), (self, other))

    def __or__(self, other: "Expr") -> "Expr":
        return Expr("OR", tuple(), (self, other))


class QuerySet(Generic[M]):
    def __init__(self, model: Type[M], filters: Optional[List[Expr]] = None):
        self._model = model
        self._filters: List[Expr] = list(filters or [])

    def where(self, *predicates: Expr) -> "QuerySet[M]":
        return QuerySet(self._model, self._filters + list(predicates))

    # Demo output: compile to pseudo-SQL + params
    def compile(self) -> Tuple[str, List[Any]]:
        if not self._filters:
            return f"SELECT * FROM {self._model.__name__}", []
        clauses: List[str] = []
        params: List[Any] = []
        for e in self._filters:
            sql, p = compile_expr(e)
            clauses.append(sql)
            params.extend(p)
        return (
            f"SELECT * FROM {self._model.__name__} WHERE " + " AND ".join(f"({c})" for c in clauses),
            params,
        )

    # For REPL friendliness
    def __repr__(self) -> str:
        sql, params = self.compile()
        return f"<QuerySet {sql} params={params!r}>"



# ======== QuerySet, Manager, and a tiny compiler ========
class Manager(Generic[M]):
    def __init__(self, model: Type[M]):
        self.model = model

    def where(self, *predicates: Expr) -> QuerySet[M]:
        return QuerySet(self.model).where(*predicates)



# ======== Descriptors for fields and relations ========
class FieldBase:
    """Common features for Field and ForeignKey."""
    model: Type[Model]
    name: str

    def __set_name__(self, owner: Type[Model], name: str):
        self.model = owner
        self.name = name

# ======== Model metaclass & base ========
class ModelMeta(type):
    def __new__(mcls, name, bases, ns):
        cls = super().__new__(mcls, name, bases, ns)

        # Attach a manager
        cls.objects = Manager(cls)

        # Bind descriptor backrefs for fields/relations
        for k, v in ns.items():
            if isinstance(v, FieldBase):
                v.__set_name__(cls, k)

        return cls


class Model(metaclass=ModelMeta):
    """Base model class (empty for this prototype)."""
    objects: "Manager[Any]"

class Field(Generic[T], FieldBase):
    """Scalar field descriptor (int, str, etc.)."""

    def __init__(self, py_type: Optional[Type[T]] = None):
        self.py_type = py_type

    def __get__(self, instance: Optional[Model], owner: Type[Model]):
        # Class access → return a FieldRef proxy
        if instance is None:
            return make_field_ref(owner, (self.name,), self.py_type)
        # Instance access (not implemented in this toy example)
        raise AttributeError("Instance access not implemented in this prototype.")


class ForeignKey(Generic[M_co], FieldBase):
    """Foreign key descriptor."""

    def __init__(self, target_model: Type[M_co]):
        self._target_model: Type[M_co] = target_model

    def __get__(self, instance: Optional[Model], owner: Type[Model]):
        if instance is None:
            return RelationRef(owner, (self.name,), self._target_model)
        raise AttributeError("Instance access not implemented in this prototype.")


# ======== Proxies returned by descriptors ========
class BaseRef:
    """Holds the model the path is rooted in and the accumulated path."""
    def __init__(self, model: Type[Model], path: Tuple[str, ...]):
        self._model = model
        self._path = path


class FieldRef(Generic[T], BaseRef):
    """Generic field proxy. Subclasses can add type-specific lookups."""

    # ----- shared lookups -----
    def exact(self, value: Optional[T]) -> Expr:
        return Expr("exact", self._path, value)

    def in_(self, values: Iterable[T]) -> Expr:
        return Expr("in", self._path, list(values))

    def isnull(self, flag: bool) -> Expr:
        return Expr("isnull", self._path, flag)

    # ----- operators (for numeric/comparable types) -----
    def __eq__(self, other: object) -> Expr:        # type: ignore[override]
        return Expr("exact", self._path, other)

    def __ne__(self, other: object) -> Expr:        # type: ignore[override]
        return Expr("ne", self._path, other)

    def __gt__(self, other: T) -> Expr:
        return Expr("gt", self._path, other)

    def __ge__(self, other: T) -> Expr:
        return Expr("gte", self._path, other)

    def __lt__(self, other: T) -> Expr:
        return Expr("lt", self._path, other)

    def __le__(self, other: T) -> Expr:
        return Expr("lte", self._path, other)


class StringFieldRef(FieldRef[str]):
    # string-specific
    def contains(self, value: str) -> Expr:
        return Expr("contains", self._path, value)

    def icontains(self, value: str) -> Expr:
        return Expr("icontains", self._path, value)

    def startswith(self, value: str) -> Expr:
        return Expr("startswith", self._path, value)

    def istartswith(self, value: str) -> Expr:
        return Expr("istartswith", self._path, value)

    def endswith(self, value: str) -> Expr:
        return Expr("endswith", self._path, value)

    def iendswith(self, value: str) -> Expr:
        return Expr("iendswith", self._path, value)


class IntFieldRef(FieldRef[int]):
    pass  # numeric comparisons already on FieldRef


class BoolFieldRef(FieldRef[bool]):
    pass


class RelationRef(Generic[M_co], BaseRef):
    """Proxy for stepping into related model's attributes (fields/relations)."""

    def __init__(self, model: Type[Model], path: Tuple[str, ...], target_model: Type[M_co]):
        super().__init__(model, path)
        self._target_model = target_model

    def __getattr__(self, name: str):
        # Grab the descriptor from the target model WITHOUT triggering descriptor protocol
        try:
            desc = self._target_model.__dict__[name]
        except KeyError as e:
            raise AttributeError(f"{self._target_model.__name__} has no attribute {name!r}") from e

        # Now use the descriptor to get a proxy rooted at the target model
        prox = desc.__get__(None, self._target_model)  # FieldRef or RelationRef
        # Re-root that proxy through our current relation path
        if isinstance(prox, FieldRef):
            # choose subclass based on prox type if available
            return reroot_fieldref(prox, self._model, self._path + prox._path)
        if isinstance(prox, RelationRef):
            return RelationRef(self._model, self._path + prox._path, prox._target_model)
        # If someone put a non-descriptor value there:
        raise AttributeError(f"Attribute {name!r} is not a field/relation on {self._target_model.__name__}")


def make_field_ref(root_model: Type[Model], path: Tuple[str, ...], py_type: Optional[Type[Any]]) -> FieldRef[Any]:
    """Factory that picks the *right* proxy subclass for completion."""
    if py_type is str:
        return StringFieldRef(root_model, path)  # type: ignore[return-value]
    if py_type is int:
        return IntFieldRef(root_model, path)     # type: ignore[return-value]
    if py_type is bool:
        return BoolFieldRef(root_model, path)    # type: ignore[return-value]
    return FieldRef(root_model, path)


def reroot_fieldref(ref: FieldRef[Any], new_root: Type[Model], new_path: Tuple[str, ...]) -> FieldRef[Any]:
    """Preserve the subclass when we re-root a field proxy through a relation chain."""
    cls = ref.__class__
    new_ref = cls.__new__(cls)  # type: ignore[misc]
    BaseRef.__init__(new_ref, new_root, new_path)  # re-init BaseRef bits
    return new_ref




# ======== Compiler helpers ========
def compile_expr(e: Expr) -> Tuple[str, List[Any]]:
    if e.op in {"AND", "OR"}:
        a, b = e.value  # type: ignore[misc]
        sa, pa = compile_expr(a)
        sb, pb = compile_expr(b)
        op = "AND" if e.op == "AND" else "OR"
        return f"({sa}) {op} ({sb})", pa + pb

    col = "__".join(e.path)  # in a real system you'd alias/join; this is demo
    v = e.value
    if e.op == "exact":
        return f"{col} = %s", [v]
    if e.op == "ne":
        return f"{col} <> %s", [v]
    if e.op == "gt":
        return f"{col} > %s", [v]
    if e.op == "gte":
        return f"{col} >= %s", [v]
    if e.op == "lt":
        return f"{col} < %s", [v]
    if e.op == "lte":
        return f"{col} <= %s", [v]
    if e.op == "in":
        placeholders = ", ".join(["%s"] * len(v))
        return f"{col} IN ({placeholders})", list(v)
    if e.op == "isnull":
        return (f"{col} IS NULL", []) if v else (f"{col} IS NOT NULL", [])
    if e.op in {"contains", "startswith", "endswith", "icontains", "istartswith", "iendswith"}:
        # naive pattern handling for demo
        case = "ILIKE" if e.op.startswith("i") else "LIKE"
        if "contains" in e.op:
            pat = f"%{v}%"
        elif "startswith" in e.op:
            pat = f"{v}%"
        else:  # endswith
            pat = f"%{v}"
        return f"{col} {case} %s", [pat]
    raise ValueError(f"Unknown op: {e.op}")


# ======== Example models (this is what your users would write) ========
class League(Model):
    id = Field[int](int)
    name = Field[str](str)


class Team(Model):
    id = Field[int](int)
    name = Field[str](str)
    league = ForeignKey(League)


class Result(Model):
    id = Field[int](int)
    score = Field[int](int)
    published = Field[bool](bool)


class Match(Model):
    id = Field[int](int)
    team = ForeignKey(Team)
    result = ForeignKey(Result)


# ======== Demo / quick tests ========
if __name__ == "__main__":
    # Autocomplete hotspots (open your editor and type along):
    #   Match.            → id, team, result
    #   Match.team.       → id, name, league
    #   Match.team.league.name. → contains, icontains, startswith, ...
    #   Match.result.id.  → comparison operators, in_(), exact()

    q = (
        Match.objects
        .where(Match.team.league.name.icontains("eagles"))
        .where(Match.result.id.in_([1, 2, 3]))
        .where(Match.result.published == True)      # noqa: E712
        .where(Match.result.score > 10)
    )

    print(q)  # shows pseudo-SQL + params
    sql, params = q.compile()
    print("SQL:", sql)
    print("Params:", params)

    # Boolean composition
    p1 = Match.team.name.startswith("New")
    p2 = Match.team.league.name.endswith("East")
    p3 = (p1 & p2) | (Match.result.score >= 42)
    q2 = Match.objects.where(p3)
    print(q2)
