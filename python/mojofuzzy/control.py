"""A focused Mamdani control-system API compatible with scikit-fuzzy."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ._lib import addr, f64, lib
from .defuzzify import defuzz, interp_universe
from .membership import trimf


class TermPrimitive:
    def __and__(self, other):
        return TermAggregate(self, other, "and")

    def __or__(self, other):
        return TermAggregate(self, other, "or")

    def __invert__(self):
        return TermAggregate(self, None, "not")


class Term(TermPrimitive):
    def __init__(self, label, parent, mf):
        self.label = label
        self.parent = parent
        self.mf = f64(mf)

    def __mod__(self, weight):
        return WeightedTerm(self, weight)


class TermAggregate(TermPrimitive):
    def __init__(self, term1, term2, kind):
        self.term1 = term1
        self.term2 = term2
        self.kind = kind


class WeightedTerm:
    def __init__(self, term, weight):
        self.term = term
        self.weight = float(weight)


class FuzzyVariable:
    def __init__(self, universe, label):
        self.universe = f64(universe)
        if self.universe.ndim != 1 or self.universe.size < 2:
            raise ValueError("universe must be a one-dimensional array")
        self.label = label
        self.terms = OrderedDict()

    def __getitem__(self, key):
        return self.terms[key]

    def __setitem__(self, key, membership):
        membership = f64(membership)
        if membership.shape != self.universe.shape:
            raise ValueError("membership function must match the universe")
        self.terms[key] = Term(key, self, membership)

    def automf(self, number=5, variable_type="quality", names=None, invert=False):
        if names is None and number not in (3, 5, 7):
            raise ValueError("number must be 3, 5, or 7 unless names are supplied")
        quality = ("dismal", "poor", "mediocre", "average", "decent", "good", "excellent")
        quantity = ("lowest", "lower", "low", "average", "high", "higher", "highest")
        if names is not None:
            labels = tuple(names)
            number = len(labels)
        else:
            labels = quality if variable_type.lower() == "quality" else quantity
            labels = labels[1:6:2] if number == 3 and variable_type.lower() == "quality" else (
                labels[2:5] if number == 3 else labels[1:6] if number == 5 else labels
            )
        if number < 2:
            raise ValueError("automf requires at least two names")
        centers = np.linspace(self.universe.min(), self.universe.max(), len(labels))
        if invert:
            labels = labels[::-1]
        self.terms.clear()
        for i, label in enumerate(labels):
            a = centers[max(0, i - 1)]
            b = centers[i]
            c = centers[min(len(centers) - 1, i + 1)]
            self[label] = trimf(self.universe, [a, b, c])


class Antecedent(FuzzyVariable):
    pass


class Consequent(FuzzyVariable):
    def __init__(self, universe, label, defuzzify_method="centroid"):
        super().__init__(universe, label)
        self.defuzzify_method = defuzzify_method
        self.accumulation_method = np.fmax


class Rule:
    def __init__(self, antecedent=None, consequent=None, label=None, and_func=np.fmin, or_func=np.fmax):
        self.antecedent = antecedent
        self.label = label
        self.and_func = and_func
        self.or_func = or_func
        if consequent is None:
            self.consequent = []
        elif isinstance(consequent, Term):
            self.consequent = [WeightedTerm(consequent, 1.0)]
        elif isinstance(consequent, WeightedTerm):
            self.consequent = [consequent]
        else:
            self.consequent = [
                item if isinstance(item, WeightedTerm) else WeightedTerm(item, 1.0)
                for item in consequent
            ]


class ControlSystem:
    def __init__(self, rules=None):
        self.rules = list(rules or [])
        self._version = 0

    def addrule(self, rule):
        self.rules.append(rule)
        self._version += 1

    @property
    def antecedents(self):
        found = OrderedDict()

        def visit(node):
            if isinstance(node, Term):
                if isinstance(node.parent, Antecedent):
                    found[node.parent.label] = node.parent
            elif isinstance(node, TermAggregate):
                visit(node.term1)
                if node.term2 is not None:
                    visit(node.term2)

        for rule in self.rules:
            visit(rule.antecedent)
        return iter(found.values())

    @property
    def consequents(self):
        found = OrderedDict()
        for rule in self.rules:
            for weighted in rule.consequent:
                variable = weighted.term.parent
                found[variable.label] = variable
        return iter(found.values())


class _InputAcceptor(dict):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

    def __setitem__(self, key, value):
        if self.simulation._ctrl_version != self.simulation.ctrl._version:
            self.simulation._refresh_structure()
        variables = self.simulation._input_variables
        if key not in variables:
            raise ValueError(f"Unexpected input: {key}")
        variable = variables[key]
        scalar = float(value)
        if self.simulation.clip_to_bounds:
            scalar = float(np.clip(scalar, variable.universe.min(), variable.universe.max()))
        elif scalar < variable.universe.min() or scalar > variable.universe.max():
            raise IndexError("Input value is outside the universe")
        super().__setitem__(key, scalar)


class ControlSystemSimulation:
    def __init__(
        self, control_system, clip_to_bounds=True, cache=True, flush_after_run=1000, lenient=True
    ):
        self.ctrl = control_system
        self.clip_to_bounds = clip_to_bounds
        self.cache = cache
        self.lenient = lenient
        self.input = _InputAcceptor(self)
        self.output = OrderedDict()
        self._ctrl_version = -1
        self._refresh_structure()

    def _refresh_structure(self):
        self._antecedents = tuple(self.ctrl.antecedents)
        self._consequents = tuple(self.ctrl.consequents)
        self._input_variables = {
            variable.label: variable for variable in self._antecedents
        }
        self._ctrl_version = self.ctrl._version

    def _truth(self, node, memberships, rule):
        if isinstance(node, Term):
            return memberships[(node.parent.label, node.label)]
        if node.kind == "not":
            return 1.0 - self._truth(node.term1, memberships, rule)
        left = self._truth(node.term1, memberships, rule)
        right = self._truth(node.term2, memberships, rule)
        return float(rule.and_func(left, right) if node.kind == "and" else rule.or_func(left, right))

    def compute(self):
        if self._ctrl_version != self.ctrl._version:
            self._refresh_structure()
        antecedents = self._antecedents
        missing = [variable.label for variable in antecedents if variable.label not in self.input]
        if missing:
            raise ValueError("All antecedents must have input values!")
        memberships = {}
        for variable in antecedents:
            value = self.input[variable.label]
            for label, term in variable.terms.items():
                memberships[(variable.label, label)] = float(
                    np.interp(value, variable.universe, term.mf, left=0.0, right=0.0)
                )

        cuts = {}
        for rule in self.ctrl.rules:
            strength = self._truth(rule.antecedent, memberships, rule)
            for weighted in rule.consequent:
                key = (weighted.term.parent.label, weighted.term.label)
                cuts[key] = max(cuts.get(key, 0.0), strength * weighted.weight)

        self.output = OrderedDict()
        for variable in self._consequents:
            active = [(term, cuts[(variable.label, label)])
                      for label, term in variable.terms.items()
                      if (variable.label, label) in cuts]
            if not active:
                if self.lenient:
                    continue
                raise ValueError(f"No terms have memberships for {variable.label}")
            extra = []
            for term, cut in active:
                extra.extend(interp_universe(variable.universe, term.mf, cut))
            universe = np.union1d(variable.universe, extra)
            memberships_matrix = np.vstack([
                np.interp(universe, variable.universe, term.mf, left=0.0, right=0.0)
                for term, _ in active
            ])
            strengths = f64([cut for _, cut in active])
            aggregated = np.empty(universe.size, dtype=np.float64)
            lib().msf_aggregate(
                addr(strengths), addr(memberships_matrix), addr(aggregated),
                len(active), universe.size,
            )
            if aggregated.sum() == 0:
                if self.lenient:
                    continue
                raise ValueError(f"Empty membership for {variable.label}")
            self.output[variable.label] = defuzz(
                universe, aggregated, variable.defuzzify_method
            )
