# simulator.py
from visual_grid_game import VisualGridHuntGame
from agent import GreedyGridAgent, SimpleReflexAgent, ModelBasedAgent

def run_grid_hunt(agent_cls=SimpleReflexAgent):
    env = VisualGridHuntGame()
    agent = agent_cls()

    print(f"=== IT3012 Grid Hunt Started ({agent_cls.__name__}) ===")
    while not env.is_done():
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        print(f"Facing: {env.facing} | Action: {action} | Percept: {percept} | Score: {env.score}")

    print(f"\nGame Over! Final Score: {env.score} after {env.steps} steps.")

if __name__ == "__main__":
    run_grid_hunt(SimpleReflexAgent)
