class KnowledgeBase:
    """A small propositional knowledge base using Horn-clause rules.

    Facts are plain strings. Each rule is a tuple (premises, conclusion), where
    premises is a list of fact strings and conclusion is a single fact string.
    """

    def __init__(self):
        self.facts = set()
        self.rules = []

    def tell_fact(self, fact_string):
        """Assert a fact into the knowledge base."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list, conclusion_string):
        """Add a rule: if every premise holds, the conclusion holds."""
        self.rules.append((premise_list, conclusion_string))

    def clear_facts(self):
        """Forget all facts. Rules are kept."""
        self.facts.clear()

    def forward_chain(self):
        """Derive every fact that follows from the current facts and rules.

        Repeatedly fires rules until a full pass adds nothing new (a fixed point).
        """
        new_facts_added = True
        while new_facts_added:
            new_facts_added = False
            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    # Modus Ponens: all premises known -> conclusion follows
                    if all(p in self.facts for p in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True
