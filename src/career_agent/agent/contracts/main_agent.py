"""Compatibility exports for the Main Agent contract modules.

New code should import from the focused modules. This facade keeps existing
callers stable while the contracts remain organized by responsibility.
"""

from career_agent.agent.contracts.profile import *
from career_agent.agent.contracts.candidates import *
from career_agent.agent.contracts.interactions import *
from career_agent.agent.contracts.resources import *
from career_agent.agent.contracts.task_state import *
from career_agent.agent.contracts.observations import *
from career_agent.agent.contracts.observations import _bounded_markdown
from career_agent.agent.contracts.context import *
from career_agent.agent.contracts.tools import *
from career_agent.agent.contracts.decisions import *
from career_agent.agent.contracts.projections import *
