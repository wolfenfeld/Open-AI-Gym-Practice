from Modules.Games.CartPoleGames import DQNCartPoleGame, CartPoleGame
from Modules.Games.LunarLanderGames import LunarLanderGame

if __name__ == "__main__":
    # run_cart_pole_game_dqn()

    cart_pole_game = CartPoleGame()
    cart_pole_game.run()

    dqn_cart_pole_game = DQNCartPoleGame()
    dqn_cart_pole_game.run()

    lunar_lander_game = LunarLanderGame()
    lunar_lander_game.run()
