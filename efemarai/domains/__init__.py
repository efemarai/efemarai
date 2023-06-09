import os

from efemarai.definition_checker import DefinitionChecker

try:
    from efemarai.metamorph.domain_loader import DomainLoader

    def load(filename, dataset=None):
        definition = DefinitionChecker.load_definition(filename).get("domains")

        if definition is None or not definition:
            return None

        return DomainLoader.load(definition[0], dataset)

    GeometricVariability = load(os.path.dirname(__file__) + "/geometric.yaml")
    ColorVariability = load(os.path.dirname(__file__) + "/color.yaml")
    NoiseVariability = load(os.path.dirname(__file__) + "/noise.yaml")
    TextVariability = load(os.path.dirname(__file__) + "/text.yaml")

except ImportError:
    pass
