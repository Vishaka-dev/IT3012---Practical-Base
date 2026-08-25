# simulator.py
from visual_grid_game import VisualGridHuntGame
from agent import GreedyGridAgent, SimpleReflexAgent, ModelBasedAgent, SearchAgent

def run_grid_hunt(agent_cls=SimpleReflexAgent, algo=None):
    env = VisualGridHuntGame()
    agent = agent_cls()
    if algo is not None and hasattr(agent, 'active_algo'):
        agent.active_algo = algo

    label = f"{agent_cls.__name__} [{agent.active_algo}]" if hasattr(agent, 'active_algo') else agent_cls.__name__
    print(f"=== IT3012 Grid Hunt Started ({label}) ===")
    while not env.is_done():
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        print(f"Facing: {env.facing} | Action: {action} | Percept: {percept} | Score: {env.score}")

    print(f"\nGame Over! Final Score: {env.score} after {env.steps} steps.")

if __name__ == "__main__":
    run_grid_hunt(SimpleReflexAgent)
