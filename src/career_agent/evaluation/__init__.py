"""Trajectory evaluation for the Main Agent's decision loop.

Every other test in this project scripts ``DecisionMaker.decide`` and checks
what the runtime does with the answer. That verifies the plumbing and says
nothing about the decision, which is where the system prompt lives: 11k
characters and ninety policy sentences with no coverage at all.

This package is the missing half. See ``trajectory`` for the contract and
``main_agent_scenarios`` for the catalogue.
"""
