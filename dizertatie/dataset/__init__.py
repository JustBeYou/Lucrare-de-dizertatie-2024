from dizertatie.dataset.dataset import load
from dizertatie.dataset.ro_sent import RoSent
from dizertatie.dataset.ro_text_summarization import RoTextSummarization
from dizertatie.dataset.rupert import Rupert

DATASETS = [RoSent, RoTextSummarization, Rupert]

__all__ = [DATASETS, load]
