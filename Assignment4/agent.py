# file: solve_and_agent.py

import unified_planning as up
from unified_planning.shortcuts import OneshotPlanner, Problem, Domain, Fluent, Object, Action, BoolType, RealType, UserType
from unified_planning.model import ProblemKind
from unified_planning.io import PDDLReader
from typing import Dict

# 1) Read & solve the PDDL with UnifiedPlanning
def solve_pddl(domain_file: str, problem_file: str):
    
    reader = PDDLReader()
    domain = reader.parse_domain(domain_file)
    problem = reader.parse_problem(domain, problem_file)
    planner = OneshotPlanner(name="tamer")
    result = planner.solve(problem)
    if result.status.keep_alive:
        print("Plan found:")
        for step in result.plan.actions:
            print("  ", step)
    else:
        print("No plan.")

# 2) A simple feed‑forward ReAct agent in Python
class SimpleRouterAgent:
    def __init__(self):
        
        self.llm_data = {
            "gpt4_openai": {"provider":"openai","capabilities":{"code","long_context"},"cost_per_mil":6.0},
            "textdavinci": {"provider":"openai","capabilities":{"code"},"cost_per_mil":0.02},
            "cohere_lamp": {"provider":"cohere","capabilities":{"multilingual"},"cost_per_mil":0.75},
            "cohere_xlarge": {"provider":"cohere","capabilities":{"multilingual"},"cost_per_mil":1.0},
            "local_vicuna": {"provider":"local","capabilities":{"code","safe_kids"},"cost_per_mil":0.0},
        }
        self.account_balances = {
            "acct_openai": 10.0,
            "acct_cohere": 5.0,
            "acct_local": 1.0
        }
        self.account_provider = {
            "acct_openai":"openai",
            "acct_cohere":"cohere",
            "acct_local":"local"
        }

    def route(self, request: Dict):

        candidates = []
        for llm, info in self.llm_data.items():
            if set(request["requires"]).issubset(info["capabilities"]):
                acct = next(a for a,p in self.account_provider.items() if p == info["provider"])
                cost_for_request = 0.001 * info["cost_per_mil"]
                if self.account_balances[acct] >= cost_for_request:
                    candidates.append((llm, acct, cost_for_request))
        if not candidates:
            return None
        best = min(candidates, key=lambda x: x[2])
        llm, acct, cost = best
        self.account_balances[acct] -= cost
        return {"request": request["name"], "llm": llm, "account": acct}

if __name__ == "__main__":
    solve_pddl("domain.pddl", "problem.pddl")

    agent = SimpleRouterAgent()
    for req in [{"name":"code_req","requires":["code"]},
                {"name":"translate_req","requires":["multilingual"]}]:
        assignment = agent.route(req)
        print("Routed:", assignment)