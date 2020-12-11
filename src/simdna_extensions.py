from simdna import synthetic
from collections import OrderedDict

class PairEmbeddableGenerator(synthetic.AbstractEmbeddableGenerator):
    """Embed a pair of embeddables with some separation.
        
    For working around the inane omission of `nothingInBetween` in the original
    version of this class.

    Arguments:
        emeddableGenerator1: instance of\
        :class:`.AbstractEmbeddableGenerator`. If an
        :class:`.AbstractSubstringGenerator` is provided, will be wrapped in\
        an instance of :class:`.SubstringEmbeddableGenerator`.
        embeddableGenerator2: same type information as for\
        ``embeddableGenerator1``
        separationGenerator: instance of\
        :class:`.AbstractQuantityGenerator`
        name: string, see :class:`DefaultNameMixin`
    """
    def __init__(self, embeddableGenerator1,
                 embeddableGenerator2, separationGenerator, name=None,
                 nothingInBetween=True):
        if isinstance(embeddableGenerator1, synthetic.AbstractSubstringGenerator):
            embeddableGenerator1 =\
                synthetic.SubstringEmbeddableGenerator(embeddableGenerator1)
        if (isinstance(embeddableGenerator2, synthetic.AbstractSubstringGenerator)):
            embeddableGenerator2 =\
                synthetic.SubstringEmbeddableGenerator(embeddableGenerator2)
        self.embeddableGenerator1 = embeddableGenerator1
        self.embeddableGenerator2 = embeddableGenerator2
        self.separationGenerator = separationGenerator
        self.nothingInBetween = nothingInBetween
        super(PairEmbeddableGenerator, self).__init__(name)

    def generateEmbeddable(self):
        """See superclass.
        """
        embeddable1 = self.embeddableGenerator1.generateEmbeddable()
        embeddable2 = self.embeddableGenerator2.generateEmbeddable()
        return synthetic.PairEmbeddable(
            embeddable1=embeddable1, embeddable2=embeddable2,
            separation=self.separationGenerator.generateQuantity(),
            nothingInBetween=self.nothingInBetween
        )

    def getJsonableObject(self):
        """See superclass.
        """
        return OrderedDict([
    ("class", "PairEmbeddableGenerator"),
    ("embeddableGenerator1", self.embeddableGenerator1.getJsonableObject()),
    ("embeddableenerator2", self.embeddableGenerator2.getJsonableObject()),
    ("separationGenerator", self.separationGenerator.getJsonableObject())])

