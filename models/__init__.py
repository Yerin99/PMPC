
from models.strat_blenderbot_small import Model as strat_blenderbot_small
from models.vanilla_blenderbot_small import Model as vanilla_blenderbot_small

from models.strat_dialogpt import Model as strat_dialogpt
from models.vanilla_dialogpt import Model as vanilla_dialogpt

# PMPC Model
from models.pmpc_blenderbot_small import Model as pmpc_blenderbot_small
from models.pmpc_blenderbot_small_018 import Model as pmpc_blenderbot_small_018

models = {

    'vanilla_blenderbot_small': vanilla_blenderbot_small,
    'strat_blenderbot_small': strat_blenderbot_small,

    'vanilla_dialogpt': vanilla_dialogpt,
    'strat_dialogpt': strat_dialogpt,

    # PMPC
    'pmpc_blenderbot_small': pmpc_blenderbot_small,
    'pmpc_blenderbot_small_018': pmpc_blenderbot_small_018,
}