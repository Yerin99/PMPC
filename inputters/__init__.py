
from inputters.strat import Inputter as strat
from inputters.vanilla import Inputter as vanilla
from inputters.pmpc import Inputter as pmpc


inputters = {
    'vanilla': vanilla,
    'strat': strat,
    'pmpc': pmpc,
}

