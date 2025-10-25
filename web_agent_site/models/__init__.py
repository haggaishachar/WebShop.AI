from web_agent_site.models.models import (
    HumanPolicy,
    RandomPolicy,
)
from web_agent_site.models.paper_rule_policy import (
    PaperRulePolicy,
)
from web_agent_site.models.simple_rule_policy import (
    SimpleRulePolicy,
)

# Baseline IL and RL policies (lazy import to avoid dependency issues)
try:
    from web_agent_site.models.il_policy import ILPolicy, create_il_policy
    from web_agent_site.models.rl_policy import RLPolicy, create_rl_policy
    _BASELINE_AVAILABLE = True
except ImportError:
    ILPolicy = None
    RLPolicy = None
    create_il_policy = None
    create_rl_policy = None
    _BASELINE_AVAILABLE = False


__all__ = [
    'HumanPolicy',
    'RandomPolicy',
    'PaperRulePolicy',
    'SimpleRulePolicy',
    'ILPolicy',
    'RLPolicy',
    'create_il_policy',
    'create_rl_policy',
]
